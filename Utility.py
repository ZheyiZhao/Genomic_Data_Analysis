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
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPClassifier

import csv
import plotly.plotly as py
import plotly.graph_objs as go
from pydoc import help
from scipy.stats.stats import pearsonr
from scipy import stats

genes = np.array(
    ['GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1', 'SNAI2', 'RET', 'NGFR', 'EGFR',
     'AXL'])


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
    for i in range(0, len(missing_Gene)):
        classification = np.delete(classification, i, 0)  # i-th row

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


def addGenes(input_file, classification, Genes):
    # for every pair in the dictionary
    for i in Genes:
        g = Genes[i]  # matrix that contains the data
        # update matrix classification
        # i contains the name of the gene
        classification = FeatureToLabel(classification, g, i)
    return classification


def inividualgenetolabel(class_file, gene_name, genes):
    gene_file = str("data/" + gene_name + ".txt")

    classification = np.genfromtxt(class_file, delimiter='\t', dtype=str)
    missing_label = []

    gene = np.genfromtxt(gene_file, delimiter='\t', dtype=str)
    gene = np.delete(gene, 0, 0)
    empty_cols = np.zeros((gene.shape[0], 3))
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
                found = 1
                break
    if found == 0:
        missing_label.append(k)

    # delete empty entries
    for j in range(0, len(missing_label)):
        gene = np.delete(gene, j, 0)

    np.savetxt("data/" + gene_name + "_class.txt", gene, delimiter='\t', fmt='%s')


def addLabelToGene(genes):
    class_file = "data/class_encoded.txt"
    for i in range(0, len(genes)):
        inividualgenetolabel(class_file, genes[i], genes)


def computeCorr(Genes):
    state = np.zeros((14, 7))
    cnt = 0
    for i in Genes:
        X = np.genfromtxt("data/" + i + "_class.txt", delimiter='\t', dtype=str)
        gene = X[:, 1]
        ER = X[:, 2]
        PR = X[:, 3]
        HER = X[:, 4]
        gene = preprocessing.scale(gene.astype(float))
        corr_ER, pv_ER = stats.spearmanr(gene, ER)
        corr_PR, pv_PR = stats.spearmanr(gene, PR)
        corr_HER, pv_HER = stats.spearmanr(gene, HER)
        state[cnt, :] = np.array([[cnt, corr_ER, pv_ER, corr_PR, pv_PR, corr_HER, pv_HER]])
        cnt = cnt + 1
    np.savetxt("data/statistics.txt", state, delimiter='\t', fmt='%s')


def boxPlot(df):
    pos = df[df.Class == 1]
    neg = df[df.Class == 0]

    # first boxplot pair
    ER_pos = plt.boxplot(pos, positions=[1, 2], widths=0.6)
    plt.setBoxColors(ER_pos)

    # second boxplot pair
    ER_neg = plt.boxplot(neg, positions=[4, 5], widths=0.6)
    plt.setBoxColors(ER_neg)

    # set axes limits and labels
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.ax.set_xticklabels(['ER+', 'ER-'])
    plt.ax.set_xticks([1.5, 4.5, 7.5])

    plt.savefig("images/" + i)


def scatterPlot(df, x, y, z):
    # Set style of scatterplot
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    # Create scatterplot of dataframe
    g = sns.lmplot(x,  # Horizontal axis
                   y,  # Vertical axis
                   data=df,  # Data source
                   fit_reg=False,  # Don't fix a regression line
                   hue=z,  # Set color
                   scatter_kws={"marker": "D",  # Set marker style
                                "s": 100})  # S marker size
    g.fig.get_axes()[0].set_yscale('log')
    plt.savefig("images/Scatter_" + x)


# plot
def plotCorrelation(Genes):
    for i in Genes:
        X = np.genfromtxt("data/" + i + "_class.txt", delimiter='\t', dtype=str)
        tmp = np.concatenate((X[:, 2:3], X[:, 1:2]), axis=1)
        df = pd.DataFrame(tmp, columns=['Class', i]).iloc[:, 0:2]
        df = df.astype(float)

        scatterPlot(df, i, 'Class', 'Class')


def LRGeneral(tmp, label, nf):
    train_acc = 0
    tes_acc = 0
    train_auc = 0
    test_auc = 0
    for i in range(0, 100):
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, label, test_size=0.2)

        data = np.append(tmp, label, axis=1)
        # print(X_train.shape)
        # print(X_test.shape)
        # print(Y_train.shape)
        # print(Y_test.shape)

        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        X_test = X_test.astype(float)
        X_train = X_train.astype(float)
        Y_test = Y_test.astype(float)
        Y_train = Y_train.astype(float)

        LR = LogisticRegression(penalty='l1')
        LR.fit(X_train, Y_train)
        Y_hat_t = LR.predict(X_train)

        train_acc = train_acc + np.mean(Y_hat_t == Y_train)

        Y_hat = LR.predict(X_test)
        tes_acc = tes_acc + np.mean(Y_hat == Y_test)

        fpr, tpr, threshold = metrics.roc_curve(Y_train, Y_hat_t)
        roc_auc = metrics.auc(fpr, tpr)

        train_auc = train_auc + roc_auc

        fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_hat)
        roc_auc = metrics.auc(fpr, tpr)

        test_auc = test_auc + roc_auc

        result = np.array([[train_acc, tes_acc, train_auc, test_auc]])

    print(result / 100)

    f = open('data/result.txt', 'ab')
    np.savetxt(f, result)
    f.close()

    # plot model
    plotLR(nf, LR, data)
    plotROC(Y_train, Y_hat_t, Y_test, Y_hat, "LR")


def intersection(Genes, number):
    X = np.genfromtxt("data/GCH1_class.txt", delimiter='\t', dtype=str)
    X = X[:, 0:3]
    X = np.append(X, np.zeros((X.shape[0], number - 1)), axis=1)

    cnt = 2
    for i in Genes:
        cnt = cnt + 1
        missing_Gene = []
        if (i != 'GCH1'):
            gene = Genes[i]
            print(gene)
            for j in range(0, X.shape[0]):
                found = 0
                for k in range(0, gene.shape[0]):
                    pid_i = X[j, 0]
                    pid_j = gene[k, 0]

                    # pid match
                    if pid_i[0:12] == pid_j:
                        X[j, cnt] = gene[k, 1]
                found = 1
            if found == 0:
                missing_Gene.append(j)

        # delete empty records
        for r in range(0, len(missing_Gene)):
            X = np.delete(X, r, 0)  # i-th row
    return X


def logisticRegression(nf, Genes):
    if (nf == 14):
        X = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)
        tmp = X[:, 4:19]
        label = X[:, 1:2]
        print("Logistic Regression on 14 features")
        LRGeneral(tmp, label, nf)
    elif (nf == 1):  # each
        for i in Genes:
            X = Genes[i]
            tmp = X[:, 1:2]
            label = X[:, 2:3]
            print("Logistic Regression on selected features")
            LRGeneral(tmp, label, nf)
    elif (nf > 1):  # combine


        classification = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)
        input_file = "data/class.txt"
        gene_reduced = addGenes(input_file, classification, Genes)
        np.savetxt("data/gene_reduced.txt", gene_reduced, delimiter='\t', fmt='%s')

        tmp = gene_reduced[:, 4:13]
        print(tmp.shape[0])
        label = gene_reduced[:, 1:2]
        print("Logistic Regression on " + str(nf))
        LRGeneral(tmp, label, nf)


def perfMeasure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def plotROC(Y_train, Y_hat_t, Y_test, Y_hat, str):
    fpr, tpr, threshold = metrics.roc_curve(Y_train, Y_hat_t)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='Testing_AUC = %0.2f' % roc_auc)

    fpr_train, tpr_train, threshold = metrics.roc_curve(Y_test, Y_hat)
    roc_auc = metrics.auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, 'b', label='Training_AUC = %0.2f' % roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig("images/" + str)

    plt.clf()


def MLP():
    X = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)

    tmp = X[:, 4:19]
    tmp = tmp.astype(float)

    X_train, X_test, Y_train, Y_test = train_test_split(tmp, X[:, 1:2], test_size=0.1)
    data = np.append(tmp, X[:, 1:2], axis=1)

    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    X_test = X_test.astype(float)
    X_train = X_train.astype(float)
    Y_test = Y_test.astype(float)
    Y_train = Y_train.astype(float)

    clf = MLPClassifier(hidden_layer_sizes=(2))

    clf.fit(X_train, Y_train)
    Y_hat_t = clf.predict(X_train)
    print("training accuracy: ", np.mean(Y_hat_t == Y_train))
    Y_hat = clf.predict(X_test)
    print("test accuracy: ", np.mean(Y_hat == Y_test))

    print(Y_hat)

    plotROC(Y_train, Y_hat_t, Y_test, Y_hat, "MLP")


