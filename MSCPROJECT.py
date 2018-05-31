from Utility import containsNonReserved, extractGene, FeatureToLabel, plotLR, deleteEmptyEntries, encoding, addGenes,inividualgenetolabel, computeCorr,addLabelToGene,corrGCH1EMT
from loadData import loadData
from PlotFigures import plotBox, scatterPlot, plotCorrelation, plotROC, plot_learning_curve,plotLearningCurve,plotHeatmap,plotGenes,plotCluster,plotStackBar,extractSurvivalData,plotKM,plotERPAndERN
from Learning import logisticRegression, perfMeasure,MLP,compareAlgorithms,tuneSVC,pipeLine,methylationAll,methyToLabel,methyLearning,methyPipe,featureSelection
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

#concatenateFiles()
# 1 Load data from TCGA
#loadData()

# 2 Combine genes with labels
#addLabelToGene(genes)


# 3 Read Data from files generated in the last step
GCH1 = np.genfromtxt("data/GCH1_class.txt", delimiter='\t', dtype=str)
CDH1 = np.genfromtxt("data/CDH1_class.txt", delimiter='\t', dtype=str)
CDH2 = np.genfromtxt("data/CDH2_class.txt", delimiter='\t', dtype=str)
VIM = np.genfromtxt("data/VIM_class.txt", delimiter='\t', dtype=str)
bCatenin = np.genfromtxt("data/bCatenin_class.txt", delimiter='\t', dtype=str)
ZEB1 = np.genfromtxt("data/ZEB1_class.txt", delimiter='\t', dtype=str)
ZEB2 = np.genfromtxt("data/ZEB2_class.txt", delimiter='\t', dtype=str)
TWIST1 = np.genfromtxt("data/TWIST1_class.txt", delimiter='\t', dtype=str)
SNAI1 = np.genfromtxt("data/SNAI1_class.txt", delimiter='\t', dtype=str)
SNAI2 = np.genfromtxt("data/SNAI2_class.txt", delimiter='\t', dtype=str)
RET = np.genfromtxt("data/RET_class.txt", delimiter='\t', dtype=str)
NGFR = np.genfromtxt("data/NGFR_class.txt", delimiter='\t', dtype=str)
EGFR = np.genfromtxt("data/EGFR_class.txt", delimiter='\t', dtype=str)
AXL = np.genfromtxt("data/AXL_class.txt", delimiter='\t', dtype=str)

# 4  Create dictionary for gene and matrix mapping
Genes = {'GCH1': GCH1, 'CDH1': CDH1, 'CDH2': CDH2, 'VIM': VIM, 'bCatenin': bCatenin,'ZEB1': ZEB1, 'ZEB2': ZEB2,
         'TWIST1': TWIST1, 'SNAI1': SNAI1, 'SNAI2': SNAI2, 'RET': RET, 'NGFR': NGFR, 'EGFR': EGFR, 'AXL': AXL}

classs = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)

EMT = {'CDH1':CDH1, 'CDH2': CDH2, 'VIM': VIM, 'bCatenin': bCatenin, 'ZEB1': ZEB1, 'ZEB2': ZEB2,
             'TWIST1': TWIST1, 'SNAI1': SNAI1, 'SNAI2': SNAI2}

# 5 plot figures
#computeCorr(Genes)
#plotBox(genes)
#plotHeatmap(genes)
#plotERPAndERN()
#plotGenes()
#plotCluster()
#plotStackBar(genes)
#plotKM(genes)
corrGCH1EMT(EMT)



#open('data/result.txt', 'w').close()

# 6 Regression on all genes
#nf = 14

#logisticRegression(nf, Genes, str("14 features"))

# 7 MLP
#  MLP()

# 8 Regression on individual of genes
# logisticRegression(1,Genes)

# 9 Regression on selected genes
nf = 2

RNA_seq = {'GCH1': GCH1, 'CDH1': CDH1, 'CDH2': CDH2, 'VIM': VIM}
rppa = {'bCatenin': bCatenin}
Methylation = {'ZEB1': ZEB1, 'ZEB2': ZEB2,
               'TWIST1': TWIST1, 'SNAI1': SNAI1, 'SNAI2': SNAI2}
RTK = {'RET': RET, 'NGFR': NGFR, 'EGFR': EGFR, 'AXL': AXL}

#logisticRegression(nf, RNA_seq, str("RNA_seq"))
#logisticRegression(nf, rppa, str("rppa"))
#logisticRegression(nf, Methylation, str("Methylation"))
#logisticRegression(nf, RTK, str("RTK"))


#compareAlgorithms(nf,Genes,str("Genes"))
#compareAlgorithms(nf,RNA_seq,str("RNA_seq"))
#compareAlgorithms(nf,rppa,str("rppa"))
#compareAlgorithms(nf,Methylation,str("Methylation"))
#compareAlgorithms(nf,RTK,str("RTK"))
#tuneSVC(RNA_seq)

#pipeLine(RTK)

#methylationAll()
#methyToLabel()
#methyPipe()
#featureSelection()
#methyLearning()


