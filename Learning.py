import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from PlotFigures import plotROC, plotLearningCurve
from  Utility import addGenes, addGenesFromLists
from sklearn.naive_bayes import BernoulliNB

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNetCV, LarsCV, LassoCV, LassoLarsCV, \
    RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, decomposition, datasets
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


def LRGeneral(tmp, label, nf, name):
    f = open('data/result.txt', 'ab')
    train_acc = 0
    tes_acc = 0
    train_auc = 0
    test_auc = 0
    for i in range(0, 100):
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, label, test_size=0.2)


        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        X_test = X_test.astype(float)
        X_train = X_train.astype(float)
        Y_test = Y_test.astype(float)
        Y_train = Y_train.astype(float)

        LR = LogisticRegressionCV()
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
    print("ROC ",result/100)
    np.savetxt(f, result, fmt="%2f")
    #np.savetxt(f, LR.coef_, fmt="%2f")

    f.close()
    # plot model
    # plotLR(nf, LR, data)
    plt.clf()
    plotROC(Y_train, Y_hat_t, Y_test, Y_hat, name)

    plotLearningCurve(LR, tmp, label.reshape(label.shape[0], ), name)


def logisticRegression(nf, Genes, name):
    if (nf == 14):
        X = np.genfromtxt("data/classification_gene.txt", delimiter='\t', dtype=str)
        # X
        tmp = X[:, 5:5 + len(Genes)]
        # y
        label = X[:, 1:2]
        # standarise data
        tmp = preprocessing.scale(tmp.astype(float))
        print("data size: " + str(tmp.shape[0]))

        LRGeneral(tmp, label, nf, name)

    elif (nf > 1):  # combine


        classification = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)
        gene_reduced = addGenes(classification, Genes)
        np.savetxt("data/gene_reduced.txt", gene_reduced, delimiter='\t', fmt='%s')

        tmp = gene_reduced[:, 5:5 + len(Genes)]
        label = gene_reduced[:, 1:2]
        tmp = preprocessing.scale(tmp.astype(float))

        print("data size: " + str(tmp.shape[0]))

        LRGeneral(tmp, label, nf, name)


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


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def compareAlgorithms(nf, Genes, cate):
    classification = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)
    input_file = "data/class.txt"
    gene_reduced = addGenes(input_file, classification, Genes)
    np.savetxt("data/gene_reduced.txt", gene_reduced, delimiter='\t', fmt='%s')

    tmp = gene_reduced[:, 5:5 + len(Genes)]
    label = gene_reduced[:, 1:2]
    tmp = preprocessing.scale(tmp.astype(float))

    print("data size: " + str(tmp.shape[0]))

    # LRGeneral(tmp, label, nf, name)

    lists = []
    for i in Genes:
        lists.append(i)

    # convert into catergorical data
    X = tmp
    Y = label.reshape(label.shape[0], )
    # for i in range(0,len(Genes)):
    #   col = pd.qcut(X[:,i],10,labels=False,duplicates='drop')
    #   X[:,i] = col


    # sigmoid
    # X = sigmoid(X)

    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegressionCV()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('Bayes', BernoulliNB()))
    models.append(('SVM', SVC()))

    data_size = X.shape[0]
    first = int(data_size / 30)
    second = int(first / 5)
    models.append(('MLP', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(first))))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("images/Compare_sigmoid_" + cate)
    plt.clf()


def methyPipe():
    methy = np.genfromtxt("data/methy_class.txt", delimiter='\t', dtype=str)

    tmp = methy[:, 5:methy.shape[1]]
    label = methy[:, 1:2]  # ER
    tmp = tmp.astype(float)
    label = label.astype(float)
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', SVC())])

    # result = cross_val_score(pipe,X,Y)
    # print(result)
    result = 0

    for i in range(0, 10):
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, label, test_size=0.2)
        pipe.fit(X_train, Y_train.reshape(Y_train.shape[0], ))
        y_preds = pipe.predict(X_test)
        result = result + np.mean(y_preds == Y_test.reshape(Y_test.shape[0], ))
    print(result / 10)


def pipeLine(Genes):
    classification = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)
    input_file = "data/class.txt"
    gene_reduced = addGenes(input_file, classification, Genes)
    np.savetxt("data/gene_reduced.txt", gene_reduced, delimiter='\t', fmt='%s')

    tmp = gene_reduced[:, 5:5 + len(Genes)]
    label = gene_reduced[:, 1:2]
    tmp = preprocessing.scale(tmp.astype(float))

    X = tmp
    Y = label.reshape(label.shape[0], )
    # sigmoid

    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', SVC())])

    # result = cross_val_score(pipe,X,Y)
    # print(result)
    result = 0

    for i in range(0, 10):
        X_train, X_test, Y_train, Y_test = train_test_split(tmp, label, test_size=0.2)
        pipe.fit(X_train, Y_train.reshape(Y_train.shape[0], ))
        Y_hat = pipe.predict(X_test)
        y_hat_t = pipe.predict(X_train)
        result = result + np.mean(Y_hat == Y_test.reshape(Y_test.shape[0], ))
        plotROC(Y_train, y_hat_t, Y_test, Y_hat, "methy")
    print(result / 10)


def tuneSVC(Genes):
    classification = np.genfromtxt("data/class.txt", delimiter='\t', dtype=str)
    input_file = "data/class.txt"
    gene_reduced = addGenes(input_file, classification, Genes)
    np.savetxt("data/gene_reduced.txt", gene_reduced, delimiter='\t', fmt='%s')

    tmp = gene_reduced[:, 5:5 + len(Genes)]
    label = gene_reduced[:, 1:2]
    tmp = preprocessing.scale(tmp.astype(float))

    X = tmp
    Y = label.reshape(label.shape[0], )
    # prepare configuration for cross validation test harness
    seed = 7

    degrees = [0, 1, 2]
    kernels = ['linear', 'rbf', 'poly']
    gammas = [0.1, 1, 10]
    cs = [0.1, 1, 10]
    record = np.zeros((1, 6))
    for degree in degrees:
        print(degree)
        for c in cs:
            for gamma in gammas:
                for kernel in kernels:
                    # prepare models
                    models = []
                    models.append(('SVM', SVC(kernel=kernel, gamma=gamma, degree=degree, C=c)))

                    # evaluate each model in turn
                    results = []
                    names = []
                    scoring = 'accuracy'

                    for name, model in models:
                        kfold = model_selection.KFold(n_splits=10, random_state=seed)
                        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                        results.append(cv_results)
                        names.append(name)
                        local_record = np.array([[degree, c, gamma, kernel, cv_results.mean(), cv_results.std()]])
                        record = np.append(record, local_record, axis=0)
                        # msg = "%d %s %f %f: %f (%f)" % (degree,kernel, gamma,c,cv_results.mean(), cv_results.std())
                        # print(msg)

    np.savetxt("data/tuneSVC.txt", record, delimiter='\t', fmt='%s')


def methylationAll():
    methy = np.genfromtxt("data/data_methylation_hm450.txt", delimiter='\t', dtype=str)
    na_list = []
    for i in range(1, methy.shape[0]):
        for j in range(0, methy.shape[1]):
            if (methy[i, j] == 'NA'):
                na_list.append(i)
                break

    methy = np.delete(methy, na_list, 0)
    methy = np.delete(methy, 1, 1)  # 1-th column (gene symbol)
    np.savetxt("data/methy_all.txt", methy, delimiter='\t', fmt='%s')

    print(methy.shape)


def methyToLabel():
    input_file = "data/methy_all.txt"
    class_file = "data/class.txt"
    classification = addGenesFromLists(input_file, class_file)
    np.savetxt("data/methy_class.txt", classification, delimiter='\t', fmt='%s')


def methyPCA():
    methy = np.genfromtxt("data/methy_class.txt", delimiter='\t', dtype=str)

    X = methy[:, 5:methy.shape[1]]
    Y = methy[:, 1:2]  # ER

    data_scaled = pd.DataFrame(preprocessing.scale(X[1:methy.shape[0], :]), columns=methy[0, 5:methy.shape[1]])
    Y = methy[1:Y.shape[0], 1:2]
    Y = Y.reshape(Y.shape[0], )

    # PCA
    pca = PCA(n_components=int(X.shape[0] / 100), svd_solver='full')
    pca.fit_transform(data_scaled)
    data_scaled.to_csv(r'data/pca_components.txt', header=None, index=None, mode='a', sep=' ')
    # data_scaled = data_scaled.astype(float)
    # data_scaled = data_scaled.astype(float)

    X = data_scaled.values
    X = X.astype(float)
    Y = Y.astype(float)
    return X, Y


# based on p value
def featureSelection(N):
    methy = np.genfromtxt("data/methy_class.txt", delimiter='\t', dtype=str)

    X = methy[:, 5:methy.shape[1]]
    Y = methy[:, 1:2]  # ER
    Y = methy[1:methy.shape[0], 1:2]
    Y = Y.reshape(Y.shape[0], )
    X = pd.DataFrame(preprocessing.scale(X[1:methy.shape[0], :]), columns=methy[0, 5:methy.shape[1]])

    model = SelectKBest(k=N)
    X_new = model.fit_transform(X, Y)

    selected = model.get_support()
    selected_genes = []
    delete_index = []
    selected_index = np.zeros((N, 1))
    cnt = 0
    for i in range(0, len(selected)):
        if (selected[i]):
            selected_index[cnt] = float(i)
            cnt = cnt + 1
            selected_genes.append(methy[0, i - 5])
        else:
            delete_index.append(i - 5)

    np.savetxt("data/selected_genes_methy", selected_genes, delimiter='\t', fmt='%s')
    Y = methy[1:methy.shape[0], 1:2]
    Y = Y.reshape(Y.shape[0], )
    X = np.delete(methy, delete_index, axis=1)
    X = X[1:X.shape[0], :]

    X = X.astype(float)
    Y = Y.astype(float)

    return X, Y
    # print(model.pvalues_)


def readMethy():
    methy = np.genfromtxt("data/methy_class.txt", delimiter='\t', dtype=str)

    X = methy[:, 5:methy.shape[1]]
    Y = methy[:, 1:2]  # ER
    X = pd.DataFrame(preprocessing.scale(X[1:methy.shape[0], :]), columns=methy[0, 5:methy.shape[1]])
    Y = methy[1:Y.shape[0], 1:2]
    Y = Y.reshape(Y.shape[0], )

    return X, Y


def methyLearning():
    methy = np.genfromtxt("data/methy_class.txt", delimiter='\t', dtype=str)
    # PCA
    # X,Y = methyPCA()

    # feature selection
    # N = 30
    # X,Y = featureSelection(N)

    # L1
    # With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected.
    # With Lasso, the higher the alpha parameter, the fewer features selected.
    X, Y = readMethy()
    lsvc = LinearSVC(C=0.005, penalty="l1", dual=False).fit(X, Y) #0.2
    model = SelectFromModel(lsvc, prefit=True)
    X = model.transform(X)
    print(X.shape[1])
    selected_genes = []
    selected = model.get_support()
    for i in range(0, len(selected)):
        if (selected[i]):
            selected_genes.append(methy[0, i - 5])

    np.savetxt("data/selected_genes_l1", selected_genes, delimiter='\t', fmt='%s')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegressionCV()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('Bayes', BernoulliNB()))
    models.append(('SVM', SVC()))

    data_size = X.shape[0]
    first = int(data_size / 30)
    second = int(first / 5)
    models.append(('MLP', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10))))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "training" + "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("images/Methy_Train_Compare")
    plt.clf()

    # TEST DATA



    for name, model in models:
        model_selection.check_cv()
        kfold = model_selection.KFold(n_splits=2, random_state=seed)
        cv_results = model_selection
        cv_results = model_selection.cross_val_score(model, X_test, y_test, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "test" + "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("images/Methy_Test_Compare")
    plt.clf()

    # ROC TEST
    LRGeneral(X, Y,2, "methy")