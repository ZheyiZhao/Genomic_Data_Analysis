from  Utility import containsNonReserved, extractGene, FeatureToLabel, plotLR, deleteEmptyEntries, encoding, addGenes,inividualgenetolabel, computeCorr,addLabelToGene



import _pickle as cp
import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import operator
from sklearn.model_selection import KFold
import math
import pickle
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import csv
import plotly.plotly as py
import plotly.graph_objs as go
from pydoc import help
from scipy.stats.stats import pearsonr
from scipy import stats

np.set_printoptions(threshold=np.nan)
dtype1 = np.dtype('str')
genes = np.array(
    ['GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1', 'SNAI2', 'RET', 'NGFR', 'EGFR',
     'AXL'])


def loadData():
    # I. pre-processing
    # 1 Load Data
    # brca_tcga:
    # Breast Invasive Carcinoma (TCGA, Provisional)
    # 1105 samples / 1098 patients

    # 1.1 RNA_seq





    RNA_seq = np.loadtxt("data/data_RNA_Seq_v2_expression_median.txt", dtype1)

    for i in range(0, RNA_seq.shape[0]):
        if RNA_seq[i, 0] == 'GCH1':
            GCH1 = extractGene(RNA_seq[0, :], RNA_seq[i, :])
            np.savetxt("data/GCH1.txt", GCH1, delimiter='\t', fmt='%s')
        elif RNA_seq[i, 0] == 'CDH1':
            CDH1 = extractGene(RNA_seq[0, :], RNA_seq[i, :])
            np.savetxt("data/CDH1.txt", CDH1, delimiter='\t', fmt='%s')
        elif RNA_seq[i, 0] == 'CDH2':
            CDH2 = extractGene(RNA_seq[0, :], RNA_seq[i, :])
            np.savetxt("data/CDH2.txt", CDH2, delimiter='\t', fmt='%s')
        elif RNA_seq[i, 0] == 'VIM':
            VIM = extractGene(RNA_seq[0, :], RNA_seq[i, :])
            np.savetxt("data/VIM.txt", VIM, delimiter='\t', fmt='%s')

    # 1.2 RPPA
    rppa = np.genfromtxt("data/data_rppa.txt", delimiter='\t', dtype=str)

    for i in range(0, rppa.shape[0]):
        if rppa[i, 0] == 'CTNNB1|beta-Catenin':
            bCatenin = extractGene(rppa[0, :], rppa[i, :])
            np.savetxt("data/bCatenin.txt", bCatenin, delimiter='\t', fmt='%s')
            break

    # 1.3 methylation
    methylation = np.genfromtxt("data/data_methylation_hm450.txt", delimiter='\t', dtype=str)

    for i in range(0, methylation.shape[0]):
        if methylation[i, 0] == 'ZEB1':
            ZEB1 = extractGene(methylation[0, :], methylation[i, :])
            np.savetxt("data/ZEB1.txt", ZEB1, delimiter='\t', fmt='%s')
        elif methylation[i, 0] == 'ZEB2':
            ZEB2 = extractGene(methylation[0, :], methylation[i, :])
            np.savetxt("data/ZEB2.txt", ZEB2, delimiter='\t', fmt='%s')
        elif methylation[i, 0] == 'TWIST1':
            TWIST1 = extractGene(methylation[0, :], methylation[i, :])
            np.savetxt("data/TWIST1.txt", TWIST1, delimiter='\t', fmt='%s')
        elif methylation[i, 0] == 'SNAI1':
            SNAI1 = extractGene(methylation[0, :], methylation[i, :])
            np.savetxt("data/SNAI1.txt", SNAI1, delimiter='\t', fmt='%s')
        elif methylation[i, 0] == 'SNAI2':
            SNAI2 = extractGene(methylation[0, :], methylation[i, :])
            np.savetxt("data/SNAI2.txt", SNAI2, delimiter='\t', fmt='%s')

    # 1.4 RTK
    RTKs = np.array(
        ['EGFR', 'INSULINR', 'AXL', 'HGFR', 'PDGFR', 'RET', 'ROR', 'TIE', 'NGFR', 'VEGFR', 'MUSK', 'EPHR', 'INSULINR',
         'EPHR'])
    methylation = np.genfromtxt("data/data_methylation_hm450.txt", delimiter='\t', dtype=str)

    for i in range(0, methylation.shape[0]):
        for j in range(0, RTKs.shape[0]):
            if methylation[i, 0] == RTKs[j]:
                tmp = extractGene(methylation[0, :], methylation[i, :])
                np.savetxt("data/" + RTKs[j] + ".txt", tmp, delimiter='\t', fmt='%s')

    # After execution:  Found RET NGFR EGFR AXL
    RET = np.genfromtxt("data/RET.txt", dtype=str)
    NGFR = np.loadtxt("data/NGFR.txt", dtype1)
    EGFR = np.loadtxt("data/EGFR.txt", dtype1)
    AXL = np.loadtxt("data/AXL.txt", dtype1)

    # Create dictionary for genes
    Genes = {'GCH1': GCH1, 'CDH1': CDH1, 'CDH2': CDH2, 'VIM': VIM, 'bCatenin': bCatenin, 'ZEB1': ZEB1, 'ZEB2': ZEB2,
             'TWIST1': TWIST1, 'SNAI1': SNAI1, 'SNAI2': SNAI2, 'RET': RET, 'NGFR': NGFR, 'EGFR': EGFR, 'AXL': AXL}


    # 1.5 classification
    # 1.5.1 Combine pid with three labels ER, PR, HER and store in class.txt

    patient = np.genfromtxt("data/data_bcr_clinical_data_patient.txt", delimiter='\t', dtype=str)
    classification = patient[:, 1].reshape(patient.shape[0], 1)
    for i in range(0, patient.shape[1]):
        if patient[0, i] == 'ER_STATUS_BY_IHC':
            classification = np.append(classification, patient[:, i].reshape(patient.shape[0], 1), axis=1)
        elif patient[0, i] == 'PR_STATUS_BY_IHC':
            classification = np.append(classification, patient[:, i].reshape(patient.shape[0], 1), axis=1)
        elif patient[0, i] == 'IHC_HER2':
            classification = np.append(classification, patient[:, i].reshape(patient.shape[0], 1), axis=1)
    # set NA
    for i in range(0, classification.shape[0]):
        for j in range(1, 4):
            if classification[i, j] == '[Not Available]':
                classification[i, j] = 'NA'
    # delete the 0-th row of titles
    classification = np.delete(classification, 0, 0)

    np.savetxt("data/class.txt", classification, delimiter='\t', fmt='%s')

    # add triple negative



    # encode pos and neg to binary
    index_list = ([1, 2, 3])

    file_name_input = "data/class.txt"
    file_name_output = "data/class.txt"
    classification = encoding(file_name_input, file_name_output, index_list)

    reserved = (['0', '1'])
    classification = deleteEmptyEntries(file_name_output, classification, index_list, reserved)


    #new column for TN
    empty_col = np.zeros((classification.shape[0],1))
    classification = np.append(classification, empty_col, axis=1)



    for i in range(0, classification.shape[0]):
        isTN = True
        for j in range(1, 4):
            if classification[i, j] == '1':
                isTN = False
                break
        if isTN == True:
            classification[i,4] = '1'
        else:
            classification[i,4] = '0'

    np.savetxt("data/class.txt", classification, delimiter='\t', fmt='%s')


    # 1.5.2
    # combine [pid labels] with genes
    input_file = "data/class.txt"
    classification = addGenes(input_file,classification, Genes)

    np.savetxt("data/classification_gene.txt", classification, delimiter='\t',fmt='%s')


    #