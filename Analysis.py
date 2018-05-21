from  Utility import containsNonReserved,extractGene,FeatureToLabel,plotLR,deleteEmptyEntries,encoding,addGenes,inividualgenetolabel,computeCorr,boxPlot,scatterPlot,plotCorrelation,logisticRegression,perfMeasure,plotROC,addLabelToGene,MLP
from loadData import loadData
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

# 1 Load data from TCGA
#loadData()

# 2 Combine genes with labels
#addLabelToGene(genes)

# 2 Read Data from files generated in the last step
GCH1 = np.genfromtxt("data/GCH1_class.txt",delimiter='\t', dtype=str)
CDH1 = np.genfromtxt("data/CDH1_class.txt", delimiter='\t', dtype=str)
CDH2 = np.genfromtxt("data/CDH2_class.txt", delimiter='\t',dtype=str)
VIM = np.genfromtxt("data/VIM_class.txt", delimiter='\t', dtype=str)
bCatenin = np.genfromtxt("data/bCatenin_class.txt", delimiter='\t', dtype=str)
ZEB1 = np.genfromtxt("data/ZEB1_class.txt",delimiter='\t', dtype=str)
ZEB2 = np.genfromtxt("data/ZEB2_class.txt",delimiter='\t', dtype=str)
TWIST1 = np.genfromtxt("data/TWIST1_class.txt", delimiter='\t', dtype=str)
SNAI1 = np.genfromtxt("data/SNAI1_class.txt",delimiter='\t',  dtype=str)
SNAI2 = np.genfromtxt("data/SNAI2_class.txt",delimiter='\t',dtype=str)
RET = np.genfromtxt("data/RET_class.txt",delimiter='\t',dtype=str)
NGFR = np.genfromtxt("data/NGFR_class.txt",delimiter='\t', dtype=str)
EGFR = np.genfromtxt("data/EGFR_class.txt",delimiter='\t',dtype=str)
AXL = np.genfromtxt("data/AXL_class.txt", delimiter='\t',dtype=str)



# 3 Create dictionary for gene and matrix mapping
Genes = {'GCH1': GCH1, 'CDH1': CDH1, 'CDH2': CDH2, 'VIM': VIM, 'bCatenin': bCatenin, 'ZEB1': ZEB1, 'ZEB2': ZEB2,
         'TWIST1': TWIST1, 'SNAI1': SNAI1, 'SNAI2': SNAI2, 'RET': RET, 'NGFR': NGFR, 'EGFR': EGFR, 'AXL': AXL}


classs = np.genfromtxt("data/class.txt", delimiter='\t',dtype=str)

# 5 Compute correlation
computeCorr(Genes)
plotCorrelation(Genes)



# 6 Regression on all genes
nf = 14
logisticRegression(nf,Genes)

# 7 MLP
#MLP()

# 8 Regression on individual of genes
#logisticRegression(1,Genes)

# 9 Regression on selected genes
Genes_reduced = {'GCH1': GCH1, 'RET': RET, 'NGFR': NGFR, 'EGFR': EGFR, 'AXL': AXL}


nf = 2
logisticRegression(nf,Genes_reduced)
