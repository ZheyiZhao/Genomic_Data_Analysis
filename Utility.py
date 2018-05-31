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
from lifelines.datasets import load_waltons
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
import csv
import plotly.plotly as py
import plotly.graph_objs as go
from pydoc import help
from scipy.stats.stats import pearsonr
from scipy import stats

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression,LogisticRegressionCV
from sklearn.kernel_approximation import RBFSampler



genes = np.array(
    ['GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1', 'SNAI2', 'RET', 'NGFR', 'EGFR',
     'AXL'])

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# call FeatureToLabel for each gene

def containsNonReserved(x, list):
    for i in range(0, len(list)):
        if x == list[i]:
            return False
    return True


def extractGene(pid, row):
    # reshape to 2d matrix
    row = row.reshape(row.shape[0], 1)
    # get patient id and reshape to column vector
    gene = pid.reshape(row.shape[0], 1)
    # assign gene value
    gene = np.append(gene, row, axis=1)
    return gene


def FeatureToLabel(classification, gene, name):
    # define column name for the given gene e.g. GCH1
    found = 0
    # find given gene in the gene list

    index = classification.shape[1]
    empty_col = np.zeros((classification.shape[0], 1))
    classification = np.append(classification, empty_col, axis=1)

    # number of patients available for the gene
    pnum = gene.shape[0]
    # number of patients available for labels
    pnum_class = classification.shape[0]
    # genes not found
    missing_Gene = []

    for i in range(0, pnum_class):
        found = 0
        for j in range(0, pnum):
            pid_class = classification[i, 0]
            pid = gene[j, 0]
            # pid match
            if pid[0:12] == pid_class:
                classification[i, index] = gene[j, 1]
                found = 1
        if found == 0:
            missing_Gene.append(i)

    # delete empty records
    classification = np.delete(classification, missing_Gene, 0)  # fixed to list

    return classification


def plotLR(nf, LR, data):
    if (nf == 1):
        df = pd.DataFrame(data, columns=['1', 'label'])
        df = df.astype(float)
        pos = df[df['label'] == 1].values
        neg = df[df['label'] == 0].values

        plt.plot(pos[:, 0], LR.predict(pos[:, 0].reshape(-1, 1)), '+', neg[:, 0], LR.predict(neg[:, 0].reshape(-1, 1)),
                 '*')
        xaxis = np.arange(0, 2001)
        # plt.plot(xaxis,(LR.coef_ * xaxis + LR.intercept_).reshape(2001,))
        plt.plot(xaxis, (1 / (1 + np.exp(-(LR.coef_ * xaxis + LR.intercept_)))).reshape(2001, ))
        plt.plot(xaxis, np.ones(2001) * 0.5)
        plt.savefig("images/Regression")


def deleteEmptyEntries(file_name_output, X, index_list, reserved):
    done = 0
    while done == 0:
        done = 1
        i = 0
        while i < X.shape[0]:
            for j in range(1, len(index_list) + 1):
                if containsNonReserved(X[i, j], reserved) == True:
                    X = np.delete(X, i, 0)
                    done = 0
                    break
            i = i + 1

    np.savetxt(file_name_output, X, delimiter='\t', fmt='%s')
    return X


def encoding(file_name_input, file_name_output, index_list):
    X = np.genfromtxt(file_name_input, delimiter='\t', dtype=str)

    for i in range(0, X.shape[0]):
        for index in range(1, len(index_list) + 1):
            if X[i, index] == 'Positive':
                X[i, index] = 0
            if X[i, index] == 'Negative':
                X[i, index] = 1
    np.savetxt(file_name_output, X, delimiter='\t', fmt='%s')
    return X

#from dictionary
def addGenes(classification, Genes):
    # for every pair in the dictionary
    for i in Genes:
        g = Genes[i]  # matrix that contains the data
        # update matrix classification
        # i contains the name of the gene
        classification = FeatureToLabel(classification, g, i)
    return classification


#from list
def addGenesFromLists(input_file,class_file):
    # class.txt : rows: patients, cols: labels
    # TCGA.. labels..
    classification = np.genfromtxt(class_file, delimiter='\t', dtype=str)
    # gene: rows: genes, cols:pids
    # [Hugo_Symbol TCGA]...
    # DAB1 ...
    # TNNT2 ...
    gene = np.genfromtxt(input_file, delimiter='\t', dtype=str)
    row_gene = gene.shape[0]
    col_gene = gene.shape[1]
    col_class = classification.shape[1]

    empty_row = np.zeros((1,col_class))
    classification = np.append(empty_row,classification,axis=0)
    row_class = classification.shape[0]

    empty_cols = np.zeros((row_class, row_gene - 1))
    empty_cols = empty_cols - 1000  # mark empty cells

    classification = np.append(classification,empty_cols,axis=1)

    # paste gene names

    for i in range (col_class,col_class + row_gene -1 ):
        classification[0,i] = gene[i - col_class + 1,0]

    # every patient
    empty_list = []
    for j in range (1,row_class):
        found = 0
        # match to class file
        for i in range (1,col_gene):
            pid_class = classification[j,0] #class
            pid = gene[0,i] #methy_all
            if (pid_class[0:12] == pid[0:12]):
                found = 1
                for k in range(1,row_gene):
                    classification[j,k+col_class-1] = gene[k,i]
        if found == 0:
            empty_list.append(j)

    #delete
    classification = np.delete(classification,empty_list,0)
    print(classification[0,:])
    return classification
    #print("methy class size = ",classification.shape)


def inividualgenetolabel(class_file, gene_name, genes):
    gene_file = str("data/" + gene_name + ".txt")

    classification = np.genfromtxt(class_file, delimiter='\t', dtype=str)
    missing_label = []

    gene = np.genfromtxt(gene_file, delimiter='\t', dtype=str)
    gene = np.delete(gene, 0, 0)
    empty_cols = np.zeros((gene.shape[0], 4))
    gene = np.append(gene, empty_cols, axis=1)
    found = 0

    # find labels
    for k in range(0, gene.shape[0]):
        for j in range(0, classification.shape[0]):
            pid_class = classification[j, 0]
            pid = gene[k, 0]
            # pid match
            if pid[0:12] == pid_class:
                found = 1
                gene[k, 2] = classification[j, 1]
                gene[k, 3] = classification[j, 2]
                gene[k, 4] = classification[j, 3]
                gene[k, 5] = classification[j, 4]
                found = 1
                break
    if found == 0:
        missing_label.append(k)

    # delete empty entries
    gene = np.delete(gene, missing_label, 0) # fixed to list


    title = gene[0, 0]
    if title[0:5] != 'TCGA':
        gene = np.delete(gene, 0, 0)
    np.savetxt("data/" + gene_name + "_class.txt", gene, delimiter='\t', fmt='%s')

# gene value -> gene + labels
def addLabelToGene(genes):
    class_file = "data/class.txt"
    for i in range(0, len(genes)):
        inividualgenetolabel(class_file, genes[i], genes)

def findBestK(X,k):
    print(X)
    X = np.sort(X)

    return X[:k]


def LOG2(x):
    for i in range(0,x.shape[0]):
        if (x[i]>0):
            x[i] = np.log2(x[i])
        elif (x[i]<0):
            x[i] = np.log2(-x[i])


    return x
def reject_outliers(data, m=2):
    data = data[abs(data - np.mean(data)) < m * np.std(data)]
    return data.reshape(data.shape[0],1)
def medianCenter(X):
    me = np.median(X)
    return X - me


#GCH1 with CDH1, CDH2, Vimentin, SNAI1, SNA2, Twist1, ZEB1/2 and b-Catenin
def corrGCH1EMT(EMT):


    gene_stat = []
    # gene data have different size
    # choose high expression (for better correlation?)
    cnt = 6
    for i in EMT:

        GCH1 = np.genfromtxt("data/GCH1.txt", delimiter='\t', dtype=str)
        X = np.genfromtxt("data/" + i +".txt", delimiter='\t', dtype=str)

        GCH1 = GCH1[:,1:2]
        X = X[:,1:2]
        GCH1 = GCH1[1:GCH1.shape[0], :]
        GCH1 = GCH1.reshape(GCH1.shape[0],1)
        X = X[1:X.shape[0], :]
        X = X.reshape(X.shape[0],1)
        GCH1 = GCH1.astype(float)
        X = X.astype(float)


        # size
        GCH1_size = GCH1.shape[0]
        gene_size = X.shape[0]

        if (GCH1_size < gene_size):
            X = X[0:GCH1_size, :]
        else:
            GCH1 = GCH1[0:gene_size, :]

        ones = np.ones((X.shape[0],1))
        tmp = np.append(ones,X,axis=1)
        tmp = np.append(tmp,X*X,axis=1)
        tmp = np.append(tmp,X*X*X,axis=1)
        tmp = np.append(tmp,X*X*X*X,axis=1)

        #X = tmp


        rbf_feature = RBFSampler(gamma=0.1, random_state=1)
        X = rbf_feature.fit_transform(X)
        #print(rbf_feature.get_params())
        Ntrain = int(GCH1.shape[0]*0.8)
        N = GCH1.shape[0]

       # GCH1 = preprocessing.scale(GCH1)
       # X = preprocessing.scale(X)

        lr = LinearRegression()
        y = GCH1[0:Ntrain,:].reshape(Ntrain,).astype('int')
        lr.fit(X[0:Ntrain:],y)
        yhat = lr.predict(X[Ntrain:N,:])
        ytest = GCH1[Ntrain:N,:].reshape(N-Ntrain,).astype('int')


         # compare size
    

        X = lr.predict(X)





        corr, pv = stats.spearmanr(GCH1, X)
        print(i,corr,pv)
        #GCH1 = sigmoid(GCH1)
        #X = sigmoid(X)

        #print("mean square error (percentage)",np.mean((yhat-ytest)**2)/np.mean(ytest))

        #GCH1 = GCH1[np.random.randint(GCH1.shape[0], size=500), :]
        #X = X[np.random.randint(X.shape[0], size=500 ), :]
        np.savetxt("data/gene_statistics.txt", gene_stat, delimiter='\t', fmt='%s')



        GCH1 = reject_outliers(GCH1)
        X = reject_outliers(X)
        GCH1_size = GCH1.shape[0]
        gene_size = X.shape[0]
        if (GCH1_size < gene_size):
            X = X[0:GCH1_size, :]
        else:
            GCH1 = GCH1[0:gene_size, :]


        # log2 median-centered intensity
        GCH1 = LOG2(GCH1)
        X = LOG2(X)
        GCH1 = medianCenter(GCH1)
        X = medianCenter(X)


        corr = round(corr,4)
        pv = round(pv,4)
        gene_stat.append([i,corr,pv])
        plt.scatter(X,GCH1)
        plt.title("r=" + str(corr) + " p=" + str(pv))
        #ax.set_title = ('r=%4f, p=%4f',corr,pv)
        plt.ylabel("gene_GCH1")
        plt.xlabel("gene_" + i)
        plt.legend()
        plt.savefig("images/GCH1-" + i)
        plt.clf()






def computeCorr(Genes):
    state = np.zeros((14, 9))
    cnt = 0
    for i in Genes:
        X = np.genfromtxt("data/" + i + "_class.txt", delimiter='\t', dtype=str)
        gene = X[:, 1]
        ER = X[:, 2]
        PR = X[:, 3]
        HER2 = X[:, 4]
        TN = X[:, 5]
        gene = preprocessing.scale(gene.astype(float))
        corr_ER, pv_ER = stats.spearmanr(gene, ER)
        corr_PR, pv_PR = stats.spearmanr(gene, PR)
        corr_HER2, pv_HER2 = stats.spearmanr(gene, HER2)
        corr_TN, pv_TN = stats.spearmanr(gene, TN)

        state[cnt, :] = np.array([[cnt, corr_ER, pv_ER, corr_PR, pv_PR, corr_HER2, pv_HER2, corr_TN, pv_TN]])
        cnt = cnt + 1
    np.savetxt("data/statistics.txt", state, delimiter='\t', fmt='%s')










