import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from data_loader_new import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace, pad_batch_v2_val

import sys
sys.path.append("..")
from VITA_model import VITA
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result
from recommend_gumbel import eval, test

torch.manual_seed(1203)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'VITA_mimic_iii'  # VITA_mimic_iv
past_name = 'past'
resume_path = 'saved/VITA_mimic_iii/08Epoch_35_JA_0.5223_DDI_0.07816_LOSS_1.2399385026098753.model'

"""
saved best epoch
# 'saved/VITA_mimic_iii_/08Epoch_35_JA_0.5223_DDI_0.07816_LOSS_1.2399385026098753.model'
# 'saved/VITA_mimic_iv/Epoch_52_JA_0.5207_DDI_0.09055_LOSS_1.3179462606297943.model'
"""
print(model_name)

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')

args = parser.parse_args()

def main(args):
    # load data
    data_path = '../data/mimic-iii/records_final.pkl' # '../data/mimic-iv/records_final2.pkl'
    voc_path =  '../data/mimic-iii/voc_final.pkl' # '../data/mimic-iv/voc_final2.pkl'
    ehr_adj_path ='../data/mimic-iii/ehr_adj_final.pkl' #'../data/mimic-iv/ehr_adj_final2.pkl'
    ddi_adj_path = '../data/mimic-iii/ddi_A_final.pkl' #'../data/mimic-iv/ddi_A_final2.pkl' 
    ddi_mask_path = '../data/mimic-iii/ddi_mask_H.pkl' #'../data/mimic-iv/ddi_mask_H2.pkl'
    device = torch.device('cuda')
    print(device)

    data = dill.load(open(data_path, 'rb'))

    data = [x for x in data if len(x) >= 2] ### Only admissions more than twice as input ####
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications

    ## data split
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = VITA(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.emb_dim, device=device)
    
    if args.Test:
        ddi_rate_list_t, ja_list_t, prauc_list_t, avg_p_list_t, avg_r_list_t, avg_f1_list_t, avg_med_list_t = [],[],[],[],[],[],[]
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        all_people = []
        all_score = []
        people_length = []
        predicted_med = []
        for step, input in enumerate(data_test):
            step_l = []
            for idx, adm in enumerate(input):
                if idx == 0: 
                    pass             
                else:
                    adm_list = []
                    
                    seq_input = input[:idx+1] ## Added cumulatively
                    adm_list.append(seq_input)
                    test_dataset = mimic_data(adm_list)
                    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)

                    smm_record, ja, prauc, precision, recall, f1, med_num, gumbel_pick_index, cross_visit_scores_numpy = test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, args)
                    
                    
                    data_num = len(ja)
                    final_length = int(data_num)
                    idx_list = list(range(data_num))
                    random.shuffle(idx_list)
                    idx_list = idx_list[:final_length]
                    avg_ja = np.mean([ja[i] for i in idx_list])
                    avg_prauc = np.mean([prauc[i] for i in idx_list])
                    avg_precision = np.mean([precision[i] for i in idx_list])
                    avg_recall = np.mean([recall[i] for i in idx_list])
                    avg_f1 = np.mean([f1[i] for i in idx_list])
                    avg_med = np.mean([med_num[i] for i in idx_list])
                    cur_smm_record = [smm_record[i] for i in idx_list]
                    ddi_rate = ddi_rate_score(cur_smm_record, path= '../data/mimic-iii/ddi_A_final.pkl') # '../data/mimic-iv/ddi_A_final2.pkl' 
                    isnan_list = [np.isnan(i) for i in [ddi_rate,avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med]]
                    if True not in isnan_list:
                        result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
                        llprint('\nDDI Rate: {}, Jaccard: {}, PRAUC: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_F1: {}, AVG_MED: {}\n'.format(
                                ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
            predicted_med.append(smm_record)
            all_people.append(gumbel_pick_index)
            print(cross_visit_scores_numpy)
            print(gumbel_pick_index, len(seq_input))
            all_score.append(cross_visit_scores_numpy)
            people_length.append(len(input))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        dill.dump(all_people, open(os.path.join('saved', model_name, 'gumbel_pick.pkl'), 'wb')) ## If there's no stored visit time chosen by the model, insert 0
        dill.dump(all_score,open(os.path.join('saved', model_name, 'all_score.pkl'), 'wb')) ## Save all attention scores for each visit
        dill.dump(people_length, open(os.path.join('saved', model_name, 'people_length.pkl'), 'wb')) ## Save the number of visits for each patient
        dill.dump(predicted_med, open(os.path.join('saved', model_name, 'predicted_med.pkl'), 'wb')) ## Save the predicted medications
        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 100 # 200
    temp_min = 0.5
    ANNEAL_RATE = 0.000003
    temp_max = 25
    ANNEAL_RATE2 = 0.000003

    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))
        loss_record = []
        gumble_list = []
        for step, input in enumerate(data_train):
            patient_list = []
            patient_list.append(len(input))
            for idx, adm in enumerate(input):
                if idx ==0 : pass  
                else:
                    adm_list = []
                    seq_input = input[:idx+1]
                    adm_list.append(seq_input)
                    train_dataset = mimic_data(adm_list)
                    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)
            
                    model.train()
                    for ind, data in enumerate(train_dataloader):
                        diseases, procedures, medications, seq_length, \
                        d_length_matrix, p_length_matrix, m_length_matrix, \
                            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                                dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                                    dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

                        diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    
                        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
                        dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
                        
                        stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
                        dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
                        stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
                        medications = medications.to(device)
                        m_mask_matrix = m_mask_matrix.to(device)
                        d_mask_matrix = d_mask_matrix.to(device)
                        p_mask_matrix = p_mask_matrix.to(device)
                        dec_disease_mask = dec_disease_mask.to(device)
                        stay_disease_mask = stay_disease_mask.to(device)
                        dec_proc_mask = dec_proc_mask.to(device)
                        stay_proc_mask = stay_proc_mask.to(device)
                        output_logits, count, gumbel_pick_index, cross_visit_scores_numpy = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
                        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
                        patient_list.append(gumbel_pick_index)
                        loss = F.nll_loss(predictions, labels.long())
                        optimizer.zero_grad()
                        loss_record.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        llprint('\rencoder_gumbel_training step: {} / {}'.format(step, len(data_train)))
            gumble_list.append(patient_list)
            
            # #### gumbel tau schedule ####
            if step % 100 == 0:
                #model.gumbel_tau = np.maximum(model.gumbel_tau * np.exp(-ANNEAL_RATE * step), temp_min)
                #model.att_tau = np.minimum(model.att_tau * np.exp(ANNEAL_RATE2 * step), temp_max)
                print(" New Gumbel Temperature: {}".format(model.gumbel_tau))
                print(" New Attention Temperature: {}".format(model.att_tau))
                
        # print("all_epoch_count: ",all_epoch_count)
         
        dill.dump(gumble_list, open(os.path.join('saved', model_name, past_name,'{}epoch_train_gumbel_pick.pkl'.format(epoch)), 'wb')) ## Save the indices of past visits similar to the current information selected by gumbel_softmax

        print ()
        tic2 = time.time()
        ## Start of the eval function
        ddi_rate_list, ja_list, prauc_list, avg_p_list, avg_r_list, avg_f1_list, avg_med_list = [],[],[],[],[],[],[]
        all_people = []
        for step, input in enumerate(data_eval):
            #step_l = []
            for idx, adm in enumerate(input):    
                if idx == 0:
                    pass
                else:  
                    adm_list = []
                    seq_input = input[:idx+1]
                    adm_list.append(seq_input)
                
                    eval_dataset = mimic_data(adm_list)
                    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_val, shuffle=True, pin_memory=True)
                    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, gumbel_pick_index, cross_visit_scores_numpy = eval(model, eval_dataloader, voc_size, device, TOKENS, args)

                    print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

                    history['ja'].append(ja)
                    history['ddi_rate'].append(ddi_rate)
                    history['avg_p'].append(avg_p)
                    history['avg_r'].append(avg_r)
                    history['avg_f1'].append(avg_f1)
                    history['prauc'].append(prauc)
                    history['med'].append(avg_med)
                    ddi_rate_list.append(ddi_rate)
                    ja_list.append(ja)
                    prauc_list.append(prauc)
                    avg_p_list.append(avg_p)
                    avg_r_list.append(avg_r)
                    avg_f1_list.append(avg_f1)
                    avg_med_list.append(avg_med)
        
        llprint('\n\rTrain--Epoch: %d, loss: %d' % (epoch, np.mean(loss_record)))
        
        """ Save the model for each epoch """
        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}_LOSS_{}.model'.format(epoch, np.mean(ja_list), np.mean(ddi_rate_list), np.mean(loss_record))), 'wb'))

        if best_ja < np.mean(ja_list):
            best_epoch = epoch
            best_ja = np.mean(ja_list)
        
        print ('best_jaccard: {}'.format(best_ja))
        print ('best_epoch: {}'.format(best_epoch))
        print('JA_{:.4}_DDI_{:.4}_PRAUC_{:.4}_F1_{:.4}'.format(np.mean(ja_list), np.mean(ddi_rate_list), np.mean(prauc_list), np.mean(avg_f1_list)))
        
        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main(args)
    print(model_name)