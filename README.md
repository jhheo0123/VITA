# VITA: ‘Carefully Chosen and Weighted Less’ Is Better in Medication Recommendation
This repository provides a reference implementation of VITA as described in the following paper:
> VITA: ‘Carefully Chosen and Weighted Less’ Is Better in Medication Recommendation  
> Taeri Kim, Jiho Heo, Hongil Kim, Kijung Shin, Sang-Wook Kim  
> The 38th Annual AAAI Conference on Artificial Intelligence (AAAI 2023)

### Overview of VITA
![vita](https://github.com/jhheo0123/VITA/assets/103116459/485911c3-7528-4ab5-b27e-67c5cfefe6ee)

### Authors
* Taeri Kim (taerik@hanyang.ac.kr)
* Jiho Heo (linda0123@hanyang.ac.kr)
* Hongil Kim (hong0814@hanyang.ac.kr)
* Kijung Shin (wook@hanyang.ac.kr)
* Sang-Wook Kim (kijungs@kaist.ac.kr)

### Requirements
The code has been tested running under Python 3.10.6. The required packages are as follows:
* pandas: 1.5.1
* dill: 0.3.6
* torch: 1.8.0+cu111
* rdkit: 2022.9.1
* scikit-learn: 1.1.3
* numpy: 1.23.4  
* install other packages if necessary

      pip install rdkit-pypi
      pip install scikit-learn, dill, dnc
      pip install torch
  
      pip install [xxx] # any required package if necessary, maybe do not specify the version

### Dataset Preparation
Change the path in processing_iii.py and processing_iv.py processing the data to get a complete records_final.pkl.  
For the MIMIC-III and -IV datasets, the following files are required:  
(Here, we do not share the MIMIC-III and -IV datasets due to reasons of personal privacy, maintaining research standards, and legal considerations.)  
- MIMIC-III
  - PRESCRIPTIONS.csv
  - DIAGNOSES_ICD.csv 
  - PROCEDURES_ICD.csv 
  - D_ICD_DIAGNOSES.csv
  - D_ICD_PROCEDURES.csv
- MIMIC-IV
  - prescriptions2.csv
  - diagnoses_icd2.csv
  - procedures_icd2.csv
  - atc32SMILES.pkl
  - ndc2atc_level4.csv (We provide a sample of this file due to size constraints.)
  - ndc2rxnorm_mapping.txt
  - drug-atc.csv  
  - drugbank_drugs_info.csv (We provide a sample of this file due to size constraints.)  
  - drug-DDI.csv (We provide a sample of this file due to size constraints.)  

### processing file
- run data processing file

      python processing_iii.py
      python processing_iv.py
- run ddi_mask_H.py
  
      python ddi_mask_H.py
The processed files are already stored in the directories "data/mimic-iii" and "data/mimic-iv".

