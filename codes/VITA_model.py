import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.modules.linear import Linear
# from data.processing import process_visit_lg2

from layers import SelfAttend
from layers import GraphConvolution


class VITA(nn.Module):
    """ VITA, based on two novel ideas: (1) relevant-Visit selectIon; (2) Target-aware Attention """
    def __init__(self, voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=64, device=torch.device('cpu')):
        super(VITA, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = 2
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end Two added encodings, both targeting the embedding of the drug
        self.MED_PAD_TOKEN = voc_size[2]+2      # Used for padding in the embedding matrix (all zeros)
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        
        # Concatenate in the dimension of diagnosis + surgery
        self.concat_embedding = nn.Sequential( 
            nn.Embedding(voc_size[0]+3 + voc_size[1]+3, emb_dim, self.DIAG_PAD_TOKEN + self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )
        self.linear_layer = nn.Linear(emb_dim,emb_dim)
        
        self.mlp_layer = nn.Linear(71,1) # Convert from the maximum dimension of diagnosis and surgery to 1 dimension
        
        # Layers to create inputs for Gumbel
        self.gumbel_layer1 = nn.Linear(64,1)
        self.gumbel_layer2 = nn.Linear(71,2)

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # Add padding_idx, indicating to take the zero vector
            nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # Used to encode the medication from the previous visit
        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2) # DO NOT USED

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        
        # EHR and DDI graphs for enriched medication representations
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        
        self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim*2, dropout=0.2, 
                 layer_norm_eps=1e-5)

        # do not use
        self.diag_self_attend = SelfAttend(emb_dim)
        self.dec_gru = nn.GRU(emb_dim*3, emb_dim, batch_first=True)
        self.diag_attn = nn.Linear(emb_dim*2, 1)
        self.proc_attn = nn.Linear(emb_dim*2, 1)
        self.W_diag_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_attn = nn.Linear(emb_dim, emb_dim)

        # weights
        self.Ws = nn.Linear(emb_dim*2, emb_dim)  # only used at initial stage
        self.Wo = nn.Linear(emb_dim, voc_size[2]+2)  # generate mode
        self.Wc = nn.Linear(emb_dim, emb_dim)  # copy mode

        self.W_dec = nn.Linear(emb_dim, emb_dim)
        self.W_stay = nn.Linear(emb_dim, emb_dim)

        # swtich network to calculate generate probablity
        self.W_z = nn.Linear(emb_dim, 1)

        # for make_query
        self.MLP_layer = nn.Linear(emb_dim * 2,1)
        self.MLP_layer2 = nn.Linear(71,1)
        self.MLP_layer3 = nn.Linear(emb_dim, 1)
        self.MLP_layer4 = nn.Linear(2, 1)
        
        # hyperparameter gumbel_tau and att_tau
        self.gumbel_tau = 0.6
        self.att_tau = 20
        
        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], voc_size[2], False)
        
    """ obtains the patient representation """
    def make_query(self, input_disease_embdding, medications):
        batch_size = 1
        max_visit_num = input_disease_embdding.size()[0]
        emb_dim = 64
        
        """ Utilizing the mlp layer to match the dimensions """
        input1 = self.MLP_layer3(input_disease_embdding).squeeze(-1) # [seq, total number of diagnosis/procedures]
        input2 = self.MLP_layer2(input1) #  [seq, 1] # Truly dense embedding
        current = input2[-1:, :]
        current2 = current.repeat(input2.size()[0],1)
        concat = torch.cat([input2, current2],dim = -1)
        
        """concatenates the dense representations of the past visit and current visit to provide the relevant-visit selection module"""
        concat2 = torch.sigmoid(self.MLP_layer4(concat))
        gumbel_input = torch.cat([concat2, 1 - concat2], dim = -1)
        
        """using gumbel_softmax"""
        pre_gumbel = F.gumbel_softmax(gumbel_input, tau = self.gumbel_tau, hard = True)[:, 0]
        gumbel  = torch.cat([pre_gumbel[:-1], torch.ones(1, device = self.device)])
        picked = input_disease_embdding.mul(gumbel.unsqueeze(-1).unsqueeze(-1).expand(-1, 71, 64)) # select relevant-visit representations

        visit_diag_embedding = self.mlp_layer(picked.transpose(-2,-1)).view(batch_size, max_visit_num, emb_dim) # Each visit's diagnosis + procedures combined into one for each visit
        
        """ target-aware attention for make query  and extract visit level score """
        cross_visit_scores, scores_encoder = self.calc_cross_visit_scores(visit_diag_embedding, gumbel) 
        score_emb = input_disease_embdding.mul(cross_visit_scores.unsqueeze(-1).unsqueeze(-1).expand(-1, 71, 64))
        q_t = torch.sum(score_emb, dim = 0, keepdim = True)
        
        gumbel_numpy = pre_gumbel.cpu().detach().numpy()
        gumbel_pick_index = [i+1 for i in (list(filter(lambda x: gumbel_numpy[x] == 1, range(len(gumbel_numpy)))))]
        
        """ gumbel pick visit index saved if Nan, input zero """
        if gumbel_pick_index == []:
            gumbel_pick_index = [0]
        
        return q_t, gumbel_pick_index, scores_encoder
    
    def encode(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        """ Parallel computation on both batch and seq dimensions (currently not considering time series information), and each medication sequence is still predicted in order """
        batch_size, max_visit_num, max_med_num = medications.size() 
        emb_dim = 64

        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        """ concat visit t's diagnoses and procedures """
        p_change = torch.tensor([1958]).to(device).repeat(torch.tensor(procedures).size(0)) + torch.tensor(procedures) 
        adm_1_2 = torch.cat([torch.tensor(diseases), p_change],dim = -1)
        input_disease_embdding = self.concat_embedding(adm_1_2).view(batch_size * max_visit_num, max_diag_num + max_proc_num, self.emb_dim)  # [batch, seq, max_diag_num, emb]
        
        d_p_mask_matrix = torch.cat([d_mask_matrix, p_mask_matrix], dim = -1) 
        d_enc_mask_matrix = d_p_mask_matrix.view(batch_size * max_visit_num, max_diag_num + max_proc_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_diag_num + max_proc_num,1) # [batch*seq, nhead, input_length, output_length] 
        
        d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num + max_proc_num, max_diag_num + max_proc_num) 
        
        """For each visit, create a query that reflects the relevant visit selection and target-aware attention modules"""
        queries = []
        for i in range(1, input_disease_embdding.size()[0]):
            q_t, gumbel_pick_index, cross_visit_scores = self.make_query(input_disease_embdding[:i+1, :, :], medications)
            queries.append(q_t)
        pre_queries = torch.cat(queries)
 
        input_disease_embdding = torch.cat([input_disease_embdding[:1, :, :], pre_queries])# [seq, 71, emb_dim]
        input_disease_embdding = input_disease_embdding.unsqueeze(dim=0) # [batch_size, max_visit, Maximum number of diagnoses/procedures, emb_dim]
        
        counts = 0

        """Construct a 'last_seq_medication' to represent the medication from the previous visit. For the first time, since there is no previous medication, fill it with 0 (you can fill it with anything, as it won't be used anyway"""
        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1) 

        # The m_mask_matrix matrix also needs to be shifted
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device) # Use a large negative value here to avoid taking away probability after softmax
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)  
       
        # Encode the last_seq_medication
        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,self.nhead,max_med_num,1)
        last_m_enc_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        encoded_medication = self.medication_encoder(last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim), src_mask=last_m_enc_mask) # (batch*seq, max_med_num, emb_dim)
        encoded_medication = encoded_medication.view(batch_size, max_visit_num, max_med_num, self.emb_dim)

        """enriched medication representations via the relations between medications"""
        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding  - ddi_embedding * self.inter
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)
        

        return input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, counts, gumbel_pick_index


    def decode(self, input_medications, input_disease_embedding, last_medication_embedding, last_medications, cross_visit_scores,
        d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory): 
        """input_medications: [batch_size, max_visit_num, max_med_num + 1], The beginning includes the SOS_TOKEN"""
        batch_size = input_medications.size(0)
        max_visit_num = input_medications.size(1) 
        max_med_num = input_medications.size(2)
        
        max_diag_num = input_disease_embedding.size(2) 

        input_medication_embs = self.med_embedding(input_medications).view(batch_size * max_visit_num, max_med_num, -1)
        input_medication_memory = drug_memory[input_medications].view(batch_size * max_visit_num, max_med_num, -1)

        m_self_mask = m_mask_matrix
        
        """ concat visit t's diagnoses and procedures mask matrix """
        d_p_mask_matrix = torch.cat([d_mask_matrix, p_mask_matrix], dim = -1)

        last_m_enc_mask = m_self_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        medication_self_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        
        m2d_mask_matrix = d_p_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        m2d_mask_matrix = m2d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num)
       
        """ medication level """
        dec_hidden = self.decoder(input_medication_embedding=input_medication_embs, input_medication_memory=input_medication_memory,
            input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num, max_diag_num, -1), 
            input_medication_self_mask=medication_self_mask, 
            d_mask=m2d_mask_matrix)

        score_g = self.Wo(dec_hidden) # [batch * max_visit_num, max_med_num, voc_size[2]+2]
        score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
        prob_g = F.softmax(score_g, dim=-1)
        score_c = self.medication_level(dec_hidden.view(batch_size, max_visit_num, max_med_num, -1), last_medication_embedding, last_m_mask, cross_visit_scores)
        # [batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num]
        prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(batch_size, max_visit_num * max_med_num, -1) # [batch, max_visit_num * input_med_num, voc_size[2]+2]

        # Based on the indices in last_seq_medication, add the values from score_c to score_c_to_g
        copy_source = last_medications.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)

        prob_c_to_g.scatter_add_(2, copy_source, score_c)
        prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)
        generate_prob = F.sigmoid(self.W_z(dec_hidden)).view(batch_size, max_visit_num, max_med_num, 1) # [batch, max_visit_num * input_med_num, 1]
        prob =  prob_g * generate_prob + prob_c_to_g * (1. - generate_prob)
        prob[:, 0, :, :] = prob_g[:, 0, :, :] 
     
        return torch.log(prob)

    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        """ input encoder for make query using Relevant-Visit Selection and Target-Aware Attention"""
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, count, gumbel_pick_index = self.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

        # Construct medications for the decoder, used for teacher forcing during the decoding process. Note that an additional dimension is added because an extra END_TOKEN will be generated 
        input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(device)    # [batch_size, seq, 1]
        input_medication = torch.cat([input_medication, medications], dim=2)      # [batch_size, seq, max_med_num + 1]
        m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float() # Use a large negative value here to avoid taking away probability after softmax
        m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)

        # visit-level and medication-leval
        output_logits = self.decode(input_medication, input_disease_embdding,encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory) 

        cross_visit_scores_numpy = cross_visit_scores.cpu().detach().numpy()
        return output_logits, count, gumbel_pick_index, cross_visit_scores_numpy
    
    """ calculate target-aware attention """
    def calc_cross_visit_scores(self, embedding, gumbel):
        """ embedding: (batch * visit_num * emb) """
        max_visit_num = embedding.size(1)
        batch_size = embedding.size(0)
  
        # Extract the current att value when calculating attention
        diag_keys = embedding[:, :, :] # key: past visits and current visit
        diag_query = embedding[:, -1: ,:] # query: current visit
        diag_scores = torch.bmm(self.linear_layer(diag_query), diag_keys.transpose(-2,-1)) / math.sqrt(diag_query.size(-1))  # attention weight

        diag_scores_encoder = diag_scores.squeeze(0).squeeze(0)
        diag_scores = diag_scores.squeeze(0).squeeze(0).masked_fill(gumbel == 0 ,-1e9)
        
        scores = F.softmax(diag_scores / self.att_tau, dim = -1)
        scores_encoder = F.softmax(diag_scores_encoder / self.att_tau, dim = -1)
        return scores, scores_encoder

    def medication_level(self, decode_input_hiddens, last_medications, last_m_mask, cross_visit_scores):
        """
        decode_input_hiddens: [batch_size, max_visit_num, input_med_num, emb_size]
        last_medications: [batch_size, max_visit_num, max_med_num, emb_size]
        last_m_mask: [batch_size, max_visit_num, max_med_num]
        cross_visit_scores: [batch_size, max_visit_num, max_visit_num]
        """
        max_visit_num = decode_input_hiddens.size(1)
        input_med_num = decode_input_hiddens.size(2)
        max_med_num = last_medications.size(2)
          
        copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num*input_med_num, self.emb_dim)
        attn_scores = torch.matmul(copy_query, last_medications.view(-1, max_visit_num*max_med_num, self.emb_dim).transpose(-2, -1)) / math.sqrt(self.emb_dim)
        med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
        attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

        visit_scores = cross_visit_scores.unsqueeze(0).unsqueeze(-1).repeat(1,1,max_med_num).view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
        scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
        row_scores = scores.sum(dim=-1, keepdim=True)
        scores = scores / row_scores    # [batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num]

        return scores

class MedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

    def forward(self, input_medication_embedding, input_medication_memory, input_disease_embdding, 
        input_medication_self_mask, d_mask):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        # [batch_size*visit_num, max_med_num+1, max_med_num+1]
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead, input_disease_embdding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        x = self.norm2(x + self._m2d_mha_block(x, input_disease_embdding, d_mask)) #+ self._m2p_mha_block(x, input_proc_embedding, p_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim) 
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class policy_network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(policy_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)