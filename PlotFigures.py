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
from  Utility import containsNonReserved, extractGene, FeatureToLabel, plotLR, deleteEmptyEntries, encoding, addGenes,inividualgenetolabel, computeCorr,addLabelToGene


genes = np.array(
    ['GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1', 'SNAI2', 'RET', 'NGFR', 'EGFR',
     'AXL'])



def separateLabels(df, name, i,mode):


    if name == 'ER':
        ERP = df[df.ER == '1']
        if mode == 0:
            ERP = ERP[[genes[i]]]
            ERP.columns = [name]
            ERP = preprocessing.scale(ERP.astype(float))

        elif mode == 1: #survival analysis
            ERP = ERP[[genes[i],'STATUS','MONTHS']]
            ERP.columns = [name,'STATUS','MONTHS']
            ERP[name] = preprocessing.scale(ERP[name])

        ERP = ERP.astype(float)


        ERN = df[df.ER == '0']
        if mode == 0:
            ERN = ERN[[genes[i]]]
            ERN.columns = [name]
            ERN = preprocessing.scale(ERN.astype(float))

        elif mode == 1:  # survival analysis
            ERN = ERN[[genes[i], 'STATUS', 'MONTHS']]
            ERN.columns = [name, 'STATUS', 'MONTHS']
            ERN[name] = preprocessing.scale(ERN[name])

        ERN = ERN.astype(float)

        return ERP, ERN

    elif name == 'PR':
        ERP = df[df.PR == '1']
        if mode == 0:
            ERP = ERP[[genes[i]]]
            ERP.columns = [name]
        elif mode == 1:  # survival analysis
            ERP = ERP[[genes[i], 'STATUS', 'MONTHS']]
            ERP.columns = [name, 'STATUS', 'MONTHS']
        ERP = ERP.astype(float)
        ERP = ERP.values

        ERN = df[df.PR == '0']
        if mode == 0:
            ERN = ERN[[genes[i]]]
            ERN.columns = [name]
        elif mode == 1:  # survival analysis
            ERN = ERN[[genes[i], 'STATUS', 'MONTHS']]
            ERN.columns = [name, 'STATUS', 'MONTHS']
        ERN = ERN.values
        ERN = ERN.astype(float)
        ERP = preprocessing.scale(ERP.astype(float))
        ERN = preprocessing.scale(ERN.astype(float))
        return ERP, ERN

    elif name == 'HER2':
        ERP = df[df.HER2 == '1']
        if mode == 0:
            ERP = ERP[[genes[i]]]
            ERP.columns = [name]
        elif mode == 1:  # survival analysis
            ERP = ERP[[genes[i], 'STATUS', 'MONTHS']]
            ERP.columns = [name, 'STATUS', 'MONTHS']
        ERP = ERP.astype(float)
        ERP = ERP.values

        ERN = df[df.HER2 == '0']
        if mode == 0:
            ERN = ERN[[genes[i]]]
            ERN.columns = [name]
        elif mode == 1:  # survival analysis
            ERN = ERN[[genes[i], 'STATUS', 'MONTHS']]
            ERN.columns = [name, 'STATUS', 'MONTHS']
        ERN = ERN.astype(float)
        ERN = ERN.values
        ERP = preprocessing.scale(ERP.astype(float))
        ERN = preprocessing.scale(ERN.astype(float))
        return ERP, ERN
    elif name == 'TN':
        ERP = df[df.TN == '1']
        if mode == 0:
            ERP = ERP[[genes[i]]]
            ERP.columns = [name]
        elif mode == 1:  # survival analysis
            ERP = ERP[[genes[i], 'STATUS', 'MONTHS']]
            ERP.columns = [name, 'STATUS', 'MONTHS']
        ERP = ERP.astype(float)
        ERP = ERP.values

        ERN = df[df.TN == '0']
        if mode == 0:
            ERN = ERN[[genes[i]]]
            ERN.columns = [name]
        elif mode == 1:  # survival analysis
            ERN = ERN[[genes[i], 'STATUS', 'MONTHS']]
            ERN.columns = [name, 'STATUS', 'MONTHS']
        ERN = ERN.values
        ERN = ERN.astype(float)
        ERP = preprocessing.scale(ERP.astype(float))
        ERN = preprocessing.scale(ERN.astype(float))
        return ERP, ERN


def extractGeneList(genes, i):
    gene = np.genfromtxt("data/" + genes[i] + "_class.txt", delimiter='\t', dtype=str)

    df = pd.DataFrame(gene[:, 1:6], columns=[genes[i], 'ER', 'PR', 'HER2', 'TN'])

    ERP, ERN = separateLabels(df, 'ER', i,0)
    PRP, PRN = separateLabels(df, 'PR', i,0)
    HER2P, HER2N = separateLabels(df, 'HER2', i,0)
    TNP, TNN = separateLabels(df, 'TN', i,0)

    data = [ERP, ERN, PRP, PRN, HER2P, HER2N, TNP, TNN]

    return data


def plotLearningCurve(estimator, X, y, name):
    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = BernoulliNB()
    plot_learning_curve(estimator, title, X, y, ylim=(-1, 1.01))
    plt.savefig("images/learning_curve_NB " + name)

    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, X, y, (-1, 1.01))

    plt.savefig("images/learning_curve_SVM " + name)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def kappaTest(g):
    X = np.genfromtxt("data/" + g + "_class.txt", delimiter='\t', dtype=str)
    label = X[:, 2:3].astype(float).astype(int)
    gene = X[:, 1:2].astype(float)
    cohen_kappa_score(gene, label)

# GCH1 - RTK + EMT
def plotHeatmap(genes):
    state = np.genfromtxt("data/statistics.txt", delimiter='\t', dtype=str)
    state = state.astype(float)
    corr = np.append(state[:, 1:2], state[:, 3:4], axis=1)
    corr = np.append(corr, state[:, 5:6], axis=1)
    corr = np.append(corr, state[:, 7:8], axis=1)

    cmap = sns.diverging_palette(128, 240, n=10)
    heatplot = sns.heatmap(corr, xticklabels=['ER', 'PR', 'HER2', 'TN'], yticklabels=genes, cmap=cmap)
    heatplot.figure.savefig("images/correlation.png")

    plt.clf()
    pv = np.append(state[:, 0:1], state[:, 2:3], axis=1)
    pv = np.append(pv, state[:, 4:5], axis=1)
    pv = np.append(pv, state[:, 6:7], axis=1)
    pv = np.append(pv, state[:, 8:9], axis=1)

    heatplot2 = sns.heatmap(pv, xticklabels=['ER', 'PR', 'HER2', 'TN'], yticklabels=genes, cmap=cmap)
    heatplot2.figure.savefig("images/pvalue.png")


def plotERPAndERN():
    state = np.genfromtxt("data/statistics.txt", delimiter='\t', dtype=str)
    state = state.astype(float)

    avg = (state[:, 3:4] + state[:, 5:6] + state[:, 7:8]) / 3
    corr = np.append(state[:,1:2],avg, axis=1)

    cmap = sns.diverging_palette(220, 20, sep=50, as_cmap=True)
    heatplot = sns.heatmap(corr, xticklabels=['ER+', 'ER-'], yticklabels=genes, cmap=cmap)
    heatplot.xaxis.tick_top()
    heatplot.figure.savefig("images/correlation_ERP_ERN.png")

    plt.clf()
    cmap2 = sns.light_palette((210, 90, 60), input="husl")
    avg = (state[:, 4:5] + state[:, 6:7] + state[:, 8:9]) / 3
    pv = np.append( state[:, 2:3], avg, axis=1)

    heatplot2 = sns.heatmap(pv, xticklabels=['ER+', 'ER-'], yticklabels=genes, cmap=cmap2)
    heatplot2.xaxis.tick_top()
    heatplot2.figure.savefig("images/pvalue_ERP_ERN.png")



def plotGenes():
    classification = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)
    classification = classification[:, 1:19]
    classification = classification.astype(float)
    classification = preprocessing.scale(classification)

    cmap = sns.diverging_palette(128, 240, n=10)
    xlabel = ['ER', 'PR', 'HER2', 'TN', 'GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1',
              'SNAI2', 'RET', 'NGFR', 'EGFR',
              'AXL']
    classification = np.sort(classification, 0)
    heatplot = sns.heatmap(classification.T, yticklabels=xlabel, xticklabels=False, cmap="coolwarm")
    heatplot.figure.savefig("images/gene.png")


def plotCluster():
    classification = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)
    classification = classification[:, 5:19]
    classification = classification.astype(float)
    classification = preprocessing.scale(classification)

    xlabel = ['GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1', 'ZEB2', 'TWIST1', 'SNAI1',
              'SNAI2', 'RET', 'NGFR', 'EGFR',
              'AXL']
    g = sns.clustermap(classification.T, yticklabels=xlabel, xticklabels=False, cmap="coolwarm")
    plt.savefig("images/cluster.png")


def percentageHigher(medians, ERP):
    cntH = 0
    for i in range(0, ERP.shape[0]):
        if (ERP[i, 0] > medians):
            cntH = cntH + 1

    return cntH / (ERP.shape[0])


def plotStackBar(genes):
    N = 8
    plt.clf()
    for i in range(0, len(genes)):
        gene = np.genfromtxt("data/" + genes[i] + "_class.txt", delimiter='\t', dtype=str)
        df = pd.DataFrame(gene[:, 1:6], columns=[genes[i], 'ER', 'PR', 'HER2', 'TN'])

        ERP, ERN = separateLabels(df, 'ER', i,0)
        PRP, PRN = separateLabels(df, 'PR', i,0)
        HER2P, HER2N = separateLabels(df, 'HER2', i,0)
        TNP, TNN = separateLabels(df, 'TN', i,0)

        gene_value = df[genes[i]].astype(float)
        gene_value = preprocessing.scale(gene_value)
        # using mean as the threshold
        # can be median or a constant
        medians = np.mean(gene_value)
        ERPH = percentageHigher(medians, ERP)
        ERNH = percentageHigher(medians, ERN)

        PRPH = percentageHigher(medians, PRP)
        PRNH = percentageHigher(medians, PRN)

        HER2PH = percentageHigher(medians, HER2P)
        HER2NH = percentageHigher(medians, HER2N)

        TNPH = percentageHigher(medians, TNP)
        TNNH = percentageHigher(medians, TNN)

        highMedians = (ERPH, ERNH, PRPH, PRNH, HER2PH, HER2NH, TNPH, TNNH)
        lowMedians = (1 - ERPH, 1 - ERNH, 1 - PRPH, 1 - PRNH, 1 - HER2PH, 1 - HER2NH, 1 - TNPH, 1 - TNNH)

        width = 0.3
        ind = np.arange(N)

        p1 = plt.bar(ind, highMedians, width)
        p2 = plt.bar(ind, lowMedians, width, bottom=highMedians)
        plt.xticks(ind, ('ER+', 'ER-', 'PR+', 'PR-', 'HER2+', 'HER2-', 'TN+', 'TN-'))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.legend((p1[0], p2[0]), ('High', 'Low'))
        plt.savefig("images/stackPlot_" + genes[i])
        plt.clf()


def extractSurvivalData():
    patient = np.genfromtxt("data/data_bcr_clinical_data_patient.txt", delimiter='\t', dtype=str)
    survival = patient[:, 1].reshape(patient.shape[0], 1)
    for i in range(0, patient.shape[1]):
        if patient[0, i] == 'OS_STATUS':
            survival = np.append(survival, patient[:, i].reshape(patient.shape[0], 1), axis=1)
        elif patient[0, i] == 'OS_MONTHS':
            survival = np.append(survival, patient[:, i].reshape(patient.shape[0], 1), axis=1)

    # delete the titles
    survival = np.delete(survival, 0, 0) #fixed number of rows

    # encoding
    for i in range(0, survival.shape[0]):
        if survival[i, 1] == 'LIVING':
            survival[i, 1] = 0
        else:
            survival[i, 1] = 1

    classification = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)
    # append column 19 20
    empty_cols = np.zeros((classification.shape[0], 2))
    classification = np.append(classification, empty_cols, axis=1)

    # match with labels
    missing_label = []
    # find labels
    for i in range(0, classification.shape[0]):
        for j in range(0, survival.shape[0]):
            pid_class = classification[i, 0]
            pid = survival[j, 0]
            # pid match
            if pid == pid_class:
                found = 1
                classification[i, 19] = survival[i, 1]
                classification[i, 20] = survival[i, 2]
                break
    if found == 0:
        missing_label.append(i)

    # delete empty entries
    gene = np.delete(classification, missing_label, 0) #fixed delete lists

    np.savetxt("data/survival_complete.txt", classification, delimiter='\t', fmt='%s')


def separateHighandLow(df, genes, i, ERP):
    ERPH = np.ones((ERP.shape[0], 3))
    ERPL = np.ones((ERP.shape[0], 3))
    ERPH = ERPH - 1000
    ERPL = ERPL - 1000
    ERPH_cnt = 0
    ERPL_cnt = 0

    data = df[genes[i]].values
    data = data.reshape(data.shape[0], 1)
    data = data.astype(float)
    data = preprocessing.scale(data)
    means = np.mean(data)
    for i in range(0, ERP.shape[0]):
        if ERP[i, 0] > means:  # high
            ERPH[ERPH_cnt, :] = ERP[i, :]
            ERPH_cnt = ERPH_cnt + 1
        else:
            ERPL[ERPL_cnt, :] = ERP[i, :]
            ERPL_cnt = ERPL_cnt + 1

    ERPH = ERPH[0:ERPH_cnt, :]
    ERPL = ERPL[0:ERPL_cnt, :]

    return ERPH, ERPL


def plotKM(genes):
    extractSurvivalData()
    data = np.genfromtxt("data/survival_complete.txt", delimiter='\t', dtype=str)

    # df = load_waltons()  # returns a Pandas DataFrame
    # print(df)

    df = pd.DataFrame(data, columns=['id', 'ER', 'PR', 'HER2', 'TN', 'GCH1', 'CDH1', 'CDH2', 'VIM', 'bCatenin', 'ZEB1',
                                     'ZEB2', 'TWIST1', 'SNAI1',
                                     'SNAI2', 'RET', 'NGFR', 'EGFR', 'AXL', 'STATUS', 'MONTHS'])

    kmf = KaplanMeierFitter()

    for i in range(0, 14):
        # divide the complete data set into type positive and type negative (e.g. ER+ and ER-)
        # data below contain the value of the gene

        ERP, ERN = separateLabels(df, 'ER', i, 1)
        # PRP, PRN = separateLabels(df, 'PR', i, 1)
        # HER2P, HER2N = separateLabels(df, 'HER2', i,1)
        # TNP, TNN = separateLabels(df, 'TN', i,1)

        # within each type (pos/neg), divide data into high/low gene expressions

        ERPH, ERPL = separateHighandLow(df, genes, i, ERP.values)

        # KM plot
        kmf.fit(ERPH[:, 2:3].astype(float), label='pos_high')
        ax = kmf.plot()
        kmf.fit(ERPL[:, 2:3].astype(float), label='pos_low')
        kmf.plot(ax=ax)

        plt.savefig("images/kmplot_" + genes[i])
        plt.clf()

def plotROC(Y_train, Y_hat_t, Y_test, Y_hat, str):
    fpr, tpr, threshold = metrics.roc_curve(Y_train, Y_hat_t)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='Testing_AUC = %0.2f' % roc_auc)#color

    fpr_train, tpr_train, threshold = metrics.roc_curve(Y_test, Y_hat)
    roc_auc = metrics.auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, 'b', label='Training_AUC = %0.2f' % roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig("images/AOC_" + str)

    plt.clf()

def plotBox(genes):
    for i in range(0, len(genes)):
        data = extractGeneList(genes, i)

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        xlabel = ['ER+', 'ER-', 'PR+', 'PR-', 'HER2+', 'HER2-', 'TN+', 'TN-']
        ax.boxplot(data, meanline=True,showfliers=False)
        ax.set_xticklabels(xlabel, ha='right')
        plt.savefig("images/box_" + str(genes[i]))



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
        classtmp = X[:, 2:3].astype(float)
        tmp = np.concatenate((classtmp.astype(int), X[:, 1:2]), axis=1)
        df = pd.DataFrame(tmp, columns=['Class', i]).iloc[:, 0:2]

        scatterPlot(df, i, 'Class', 'Class')