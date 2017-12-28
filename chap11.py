from collections import Counter

import sklearn
from sklearn import metrics

import scipy.io as sio
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sklearn.svm import SVR
from sklearn import svm

#回归数据集的分析
#非线性SVR 核函数为高斯核 测试gamma、C值
def SVR_rbf(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    #测试C值
    C_train_mse = []
    C_test_mse = []
    C_pre_cor = []
    CC = range(1, 40)
    for C in CC:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_rbf = SVR(kernel='rbf', gamma=1, C=C)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_rbf.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_rbf.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_rbf.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        C_train_mse.append(np.mean(trainmse))#训练集的MSE
        C_test_mse.append(np.mean(testmes))#测试集的MSE
        C_pre_cor.append(pre_list.corr(d))#预测值与真实值的相关系数

    #测试gamma值
    G_train_mse = []
    G_test_mse = []
    G_pre_cor = []
    gammas = range(1, 40)
    for gamma in gammas:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_poly = SVR(kernel='rbf', gamma=gamma, C=1)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_poly.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_poly.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_poly.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        G_train_mse.append(np.mean(trainmse))#训练集的MSE
        G_test_mse.append(np.mean(testmes))#测试集的MSE
        G_pre_cor.append(pre_list.corr(d))#预测值与真实值的相关系数

    #画图
    fig = plt.figure("rbf",figsize=(25,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("SVR_rbf_C")
    ax1.plot(CC, C_test_mse, color='g', label='test_mse')
    ax1.plot(CC, C_train_mse, color='r', label='train_mse')
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("C")
    ax1.legend(loc='upper right')
    ax2 = ax1.twinx()
    ax2.plot(CC, C_pre_cor, 'o-', label='r')
    ax2.set_ylabel("PCCs")
    ax2.legend(loc='lower right')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='b')

    ax3 = fig.add_subplot(1, 2, 2)
    ax3.set_title("SVR_rbf_G")
    ax3.plot(gammas, G_test_mse, color='g', label='test_mse')
    ax3.plot(gammas, G_train_mse, color='r', label='train_mse')
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("gammas")
    ax3.legend(loc='upper right')
    ax4 = ax3.twinx()
    ax4.plot(gammas,G_pre_cor, 'o-', label='r')
    ax4.set_ylabel("PCCs")
    ax4.legend(loc='lower right')
    ax4.yaxis.label.set_color('blue')
    ax4.tick_params(axis='y', colors='b')
    plt.savefig("SVR_rbf")
    plt.show()

#非线性SVR 核函数为多项式 测试degree,gamma,ceof
def SVR_poly(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    # 测试degree值
    D_train_mse = []
    D_test_mse = []
    D_pre_cor = []
    DD = range(1, 10)
    for degree in DD:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_poly = SVR(kernel='poly',degree=degree,coef0=1)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_poly.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_poly.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_poly.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        D_train_mse.append(np.mean(trainmse))  # 训练集的MSE
        D_test_mse.append(np.mean(testmes))  # 测试集的MSE
        D_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数

    # 测试gamma值
    G_train_mse = []
    G_test_mse = []
    G_pre_cor = []
    gammas = range(1, 100)
    for gamma in gammas:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_poly = SVR(kernel='poly', gamma=gamma, degree=3,coef0=1)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_poly.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_poly.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_poly.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        G_train_mse.append(np.mean(trainmse))  # 训练集的MSE
        G_test_mse.append(np.mean(testmes))  # 测试集的MSE
        G_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
    #测试coef0值
    C_train_mse = []
    C_test_mse = []
    C_pre_cor = []
    CC = range(1, 200)
    for C in CC:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_poly = SVR(kernel='poly', gamma=40,degree=3, coef0=C)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_poly.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_poly.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_poly.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        C_train_mse.append(np.mean(trainmse))  # 训练集的MSE
        C_test_mse.append(np.mean(testmes))  # 测试集的MSE
        C_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
    # 画图
    fig = plt.figure("poly", figsize=(30, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("SVR_poly_Degree")
    ax1.plot(DD, D_test_mse, color='g', label='test_mse')
    ax1.plot(DD, D_train_mse, color='r', label='train_mse')
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("Degree")
    ax1.legend(loc='upper right')
    ax2 = ax1.twinx()
    ax2.plot(DD, D_pre_cor, '--', label='r')
    ax2.set_ylabel("PCCs")
    ax2.legend(loc='center right')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='b')

    ax3 = fig.add_subplot(1, 3, 2)
    ax3.set_title("SVR_poly_G")
    ax3.plot(gammas, G_test_mse, color='g', label='test_mse')
    ax3.plot(gammas, G_train_mse, color='r', label='train_mse')
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("lower right")
    ax3.legend(loc='best')
    ax4 = ax3.twinx()
    ax4.plot(gammas, G_pre_cor, '--', label='r')
    ax4.set_ylabel("PCCs")
    ax4.legend(loc='center right')
    ax4.yaxis.label.set_color('blue')
    ax4.tick_params(axis='y', colors='b')

    ax5 = fig.add_subplot(1, 3, 3)
    ax5.set_title("SVR_poly_Coef0")
    ax5.plot(CC, C_test_mse, color='g', label='test_mse')
    ax5.plot(CC, C_train_mse, color='r', label='train_mse')
    ax5.set_ylabel("MSE")
    ax5.set_xlabel("coef0")
    ax5.legend(loc='lower right')
    ax6 = ax5.twinx()
    ax6.plot(CC,C_pre_cor, '--', label='r')
    ax6.set_ylabel("PCCs")
    ax6.legend(loc='center right')
    ax6.yaxis.label.set_color('blue')
    ax6.tick_params(axis='y', colors='b')
    plt.savefig("SVR_poly")
    plt.show()

#非线性SVR 核函数为sigmoid 测试gamma,coef0值
def SVR_sigmoid(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    #测试gamma值
    G_train_mse = []
    G_test_mse = []
    G_pre_cor = []
    gammas = np.logspace(-1,2)
    for gamma in gammas:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_sigmoid = SVR(kernel='sigmoid', gamma=gamma, coef0=0.01)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_sigmoid.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_sigmoid.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_sigmoid.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        G_train_mse.append(np.mean(trainmse))  # 训练集的MSE
        G_test_mse.append(np.mean(testmes))  # 测试集的MSE
        G_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
    # 测试coef0值
    C_train_mse = []
    C_test_mse = []
    C_pre_cor = []
    CC = np.logspace(0,3)
    for C in CC:
        trainmse = []
        testmes = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            svr_sigmoid = SVR(kernel='sigmoid', gamma=10,coef0=C)
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_sigmoid.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_sigmoid.predict(DataBJ63_pca_train)
            trainmse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_sigmoid.predict(DataBJ63_pca_test)
            testmes.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())
        C_train_mse.append(np.mean(trainmse))  # 训练集的MSE
        C_test_mse.append(np.mean(testmes))  # 测试集的MSE
        C_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
    # 画图
    fig = plt.figure("sigmoid", figsize=(25, 5))
    ax3 = fig.add_subplot(1, 2, 1)
    ax3.set_title("SVR_sigmoid_G")
    ax3.plot(gammas, G_test_mse, color='g', label='test_mse')
    ax3.plot(gammas, G_train_mse, color='r', label='train_mse')
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("gammas")
    ax3.legend(loc='lower right')
    ax4 = ax3.twinx()
    #ax4.plot(gammas, np.absolute(G_pre_cor), 'o-', label='r')
    ax4.plot(gammas,G_pre_cor, '--', label='r')
    ax4.set_ylabel("PCCs")
    ax4.legend(loc='upper right')
    ax4.yaxis.label.set_color('blue')
    ax4.tick_params(axis='y', colors='b')

    ax5 = fig.add_subplot(1, 2, 2)
    ax5.set_title("SVR_sigmoid_Coef0")
    ax5.plot(CC, C_test_mse, color='g', label='test_mse')
    ax5.plot(CC, C_train_mse, color='r', label='train_mse')
    ax5.set_ylabel("MSE")
    ax5.set_xlabel("coef0")
    ax5.legend(loc='center right')
    ax6 = ax5.twinx()
    #ax6.plot(CC, np.absolute(C_pre_cor), 'o-', label='r')
    ax6.plot(CC, C_pre_cor, '--', label='r')
    ax6.set_ylabel("PCCs")
    ax6.legend(loc='upper right')
    ax6.yaxis.label.set_color('blue')
    ax6.tick_params(axis='y', colors='b')

    plt.savefig("SVR_sigmoid")
    plt.show()

#非线性SVR 核函数为linear
def SVR_linear(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    L1_train_mse = []
    L1_test_mse = []
    L1_pre_cor = []
    pre_list = []
    for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
        DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
        DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
        svr_linear = SVR(kernel='linear')
        svr_linear.fit(DataBJ63_pca_train, DataAgeBJ63_train)
        train_predict = svr_linear.predict(DataBJ63_pca_train)
        L1_train_mse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
        test_predict = svr_linear.predict(DataBJ63_pca_test)
        L1_test_mse.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
        pre_list.append(test_predict)
    pre_list = pd.Series(np.array(pre_list).flatten())
    meantestmse = []
    for i in range(1,64):
        meantestmse.append(np.mean(L1_test_mse))
        L1_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
    # 画图
    fig = plt.figure("linear", figsize=(25, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("SVR_linear")
    ax1.plot(range(1, 64), L1_test_mse, color='g', label='test_mse')
    ax1.plot(range(1, 64), L1_train_mse, color='r', label='train_mse')
    ax1.plot(range(1, 64), meantestmse, color='y', label='meantest_mse')
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("the num of data")
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(range(1, 64), L1_pre_cor, 'o-', label='r')
    ax2.set_ylabel("PCCs")
    ax2.legend(loc='upper right')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='b')
    plt.savefig("SVR_linear")
    plt.show()
    # print("相关系数是")
    # print(L1_pre_cor)

#线性SVR 损失函数
def LinearSVR(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    # 测试不同Loss
    losses=['squared_epsilon_insensitive','epsilon_insensitive']
    for loss in losses:
        L1_train_mse = []
        L1_test_mse = []
        L1_pre_cor = []
        pre_list = []
        for train_index, test_index in loo.split(DataBJ63_pca, DataAgeBJ63):
            DataBJ63_pca_train, DataBJ63_pca_test = DataBJ63_pca[train_index], DataBJ63_pca[test_index]
            DataAgeBJ63_train, DataAgeBJ63_test = DataAgeBJ63[train_index], DataAgeBJ63[test_index]
            svr_loss1 = sklearn.svm.LinearSVR(loss=loss)
            svr_loss1.fit(DataBJ63_pca_train, DataAgeBJ63_train)
            train_predict = svr_loss1.predict(DataBJ63_pca_train)
            L1_train_mse.append(metrics.mean_squared_error(DataAgeBJ63_train, train_predict))
            test_predict = svr_loss1.predict(DataBJ63_pca_test)
            L1_test_mse.append(metrics.mean_squared_error(DataAgeBJ63_test, test_predict))
            pre_list.append(test_predict)
        pre_list = pd.Series(np.array(pre_list).flatten())

        mean_testmse =[]
        for i in range(1,64):
            mean_testmse.append(np.mean(L1_test_mse))
            L1_pre_cor.append(pre_list.corr(d))  # 预测值与真实值的相关系数
        # 画图
        fig = plt.figure(loss, figsize=(25, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("SVR_"+loss)
        ax1.plot(range(1,64), L1_test_mse, color='g', label='test_mse')
        ax1.plot(range(1,64), L1_train_mse, color='r', label='train_mse')
        ax1.plot(range(1, 64), mean_testmse, color='y', label='mean_testmse')
        ax1.set_ylabel("MSE")
        ax1.set_xlabel("the num of data")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(range(1, 64), L1_pre_cor, 'o-', label='r')
        ax2.set_ylabel("PCCs")
        ax2.legend(loc='upper right')
        ax2.yaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='b')
        plt.savefig("LinearSVR "+loss)
        plt.show()


#比较最优参数的 不同核函数 的 MSE 和  相关系数
def SVR_All(DataBJ63_pca,DataAgeBJ63,loo):
    d = pd.Series(DataAgeBJ63.flatten())
    # 线性SVR
    linear_svr = sklearn.svm.LinearSVR(loss='epsilon_insensitive')
    # 非线性SVR linear
    svr_linear = SVR(kernel='linear')
    # 非线性SVR rbf
    svr_rbf = SVR(kernel='rbf', gamma=1,C=4)
    # 非线性SVR poly
    svr_poly = SVR(kernel='poly', gamma=40, degree=1, coef0=1)
    # 非线性SVR sigmoid
    svr_sigmoid = SVR(kernel='sigmoid', gamma=10, coef0=10)

    # 计算预测标签
    pre_linear_Label = cross_val_predict(linear_svr, DataBJ63_pca, DataAgeBJ63, cv=loo)
    pre_Label_linear = cross_val_predict(svr_linear, DataBJ63_pca, DataAgeBJ63, cv=loo)
    pre_Label_rbf = cross_val_predict(svr_rbf, DataBJ63_pca, DataAgeBJ63, cv=loo)
    pre_Label_poly = cross_val_predict(svr_poly, DataBJ63_pca, DataAgeBJ63, cv=loo)
    pre_Label_sigmoid = cross_val_predict(svr_sigmoid, DataBJ63_pca, DataAgeBJ63, cv=loo)
    # 计算MSE
    linear_MSE = metrics.mean_squared_error(DataAgeBJ63, pre_linear_Label)
    MSE_linear = metrics.mean_squared_error(DataAgeBJ63, pre_Label_linear)
    MSE_rbf = metrics.mean_squared_error(DataAgeBJ63, pre_Label_rbf)
    MSE_poly = metrics.mean_squared_error(DataAgeBJ63, pre_Label_poly)
    MSE_sigmoid = metrics.mean_squared_error(DataAgeBJ63, pre_Label_sigmoid)
    # 计算cor
    linear_COR = pd.Series(pre_linear_Label).corr(d)
    COR_linear = pd.Series(pre_Label_linear).corr(d)
    COR_rbf = pd.Series(pre_Label_rbf).corr(d)
    COR_poly = pd.Series(pre_Label_poly).corr(d)
    COR_sigmoid = pd.Series(pre_Label_sigmoid).corr(d)
    # 画图
    fig = plt.figure("SVR_ALL")
    NAMEMSE=['linear_MSE','MSE_linear','MSE_rbf','MSE_poly','MSE_sigmoid']
    XMSE = range(len(NAMEMSE))
    ALLMSE=[linear_MSE,MSE_linear,MSE_rbf,MSE_poly,MSE_sigmoid]

    NAMECOR=['linear_COR','COR_linear','COR_rbf','COR_poly','COR_sigmoid']
    XCOR = range(len(NAMECOR))
    ALLCOR=[linear_COR,COR_linear,COR_rbf,COR_poly,COR_sigmoid]
    NAME = ['linearSVR', 'SVR_linear', 'rbf', 'poly', 'sigmoid']
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title("the MSE and COR of SVR")
    ax1.set_xlabel('the kernel')
    ax1.set_ylabel('MSE')
    ax1.plot(XMSE,ALLMSE,'ro-',label='MSE')
    plt.xticks(XMSE, NAME, rotation=45)
    ax1.legend(loc='lower right')

    ax2=ax1.twinx()
    ax2.plot(XCOR, ALLCOR, 'go-', label='r')
    ax2.set_ylabel("PCCs")
    #ax2.xticks(XCOR, NAMECOR, rotation=45)
    ax2.legend(loc='upper right')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='b')

    plt.savefig("SVR_ALL")
    plt.show()
    print(ALLMSE)
    print(ALLCOR)

#SVR分析降维后的数据和标签
def SVR_Anylise(data,target):
    loo = LeaveOneOut()#使用交叉验证的LeaveOneOut
    loo.get_n_splits(data,target)
    print("SVR_rbf")
    SVR_rbf(data,target,loo)
    print("SVR_poly")
    SVR_poly(data,target,loo)
    print("SVR_sigmoid")
    SVR_sigmoid(data,target,loo)
    print("SVR_linear")
    SVR_linear(data,target,loo)
    print("LinearSVR")
    LinearSVR(data,target,loo)
    SVR_All(data, target, loo)

#通过输出降维后各主成成分的方差值所占总方差的比例决定降维数
def PCA_com(data):
    pca = PCA(n_components=0.99)#表示保留了99%的信息
    low=pca.fit_transform(data).shape[1]#降低后的维数
    Data_pca = PCA(n_components=low)
    data_pca = Data_pca.fit_transform(data)
    print("low")
    print(low)
    return data_pca

#SVC分析降维后的数据和标签
def SVC_Anylise(data,target):
    loo = LeaveOneOut()  # 使用交叉验证的LeaveOneOut
    loo.get_n_splits(data, target)
    SVC_Linear(data, target, loo)
    LinearSVC(data, target, loo)
    SVC_poly(data, target, loo)
    SVC_rbf(data,target,loo)
    SVC_sigmoid(data, target, loo)
    SVC_ALL(data,target,loo)

#比较最优参数 不同的 核函数 的 ROC 和 正确率
def SVC_ALL(DataBJ76_pca,DataLabel,loo):

    pre_linear_sc = []
    pre_sc_linear = []
    pre_sc_rbf = []
    pre_sc_poly = []
    pre_sc_sigmoid = []
    random_state = np.random.RandomState(0)
    #线性SVC
    linear_svc = sklearn.svm.LinearSVC(loss='hinge')
    #非线性SVC linear
    svc_linear = sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state)
    #非线性SVC rbf
    svc_rbf = sklearn.svm.SVC(kernel='rbf',gamma=1)
    #非线性SVC poly
    svc_poly = sklearn.svm.SVC(kernel='poly',gamma=1,degree=1,coef0=0)
    #非线性SVC sigmoid
    svc_sigmoid= sklearn.svm.SVC(kernel='sigmoid',gamma=3,coef0=0)

    # 计算预测标签
    pre_linear_Label = cross_val_predict(linear_svc,DataBJ76_pca,DataLabel,cv=loo)
    pre_Label_linear = cross_val_predict(svc_linear, DataBJ76_pca, DataLabel, cv=loo)
    pre_Label_rbf = cross_val_predict(svc_rbf, DataBJ76_pca, DataLabel, cv=loo)
    pre_Label_poly = cross_val_predict(svc_poly, DataBJ76_pca, DataLabel, cv=loo)
    pre_Label_sigmoid = cross_val_predict(svc_sigmoid, DataBJ76_pca, DataLabel, cv=loo)
    #计算正确率
    linear_Accuracy = Counter(list(pre_linear_Label  == DataLabel.flatten()))[True]/76
    Accuracy_linear = Counter(list(pre_Label_linear == DataLabel.flatten()))[True] / 76
    Accuracy_rbf = Counter(list(pre_Label_rbf == DataLabel.flatten()))[True] / 76
    Accuracy_poly = Counter(list(pre_Label_poly == DataLabel.flatten()))[True] / 76
    Accuracy_sigmoid = Counter(list(pre_Label_sigmoid == DataLabel.flatten()))[True] / 76
    #循环迭代
    for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
        linear_probas_ = linear_svc.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        probas_linear =  svc_linear.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        probas_rbf = svc_rbf.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        probas_poly = svc_poly.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        probas_sigmoid = svc_sigmoid.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        pre_linear_sc.append(linear_probas_)
        pre_sc_linear.append(probas_linear)
        pre_sc_rbf.append(probas_rbf)
        pre_sc_poly.append(probas_poly)
        pre_sc_sigmoid.append(probas_sigmoid)
    linear_fpr, linear_tpr, linear_thresholds = roc_curve(DataLabel, np.array(pre_linear_sc).flatten(), pos_label=1)
    fpr_linear, tpr_linear, thresholds_linear = roc_curve(DataLabel, np.array(pre_sc_linear).flatten(), pos_label=1)
    fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(DataLabel, np.array(pre_sc_rbf).flatten(), pos_label=1)
    fpr_poly, tpr_poly, thresholds_poly = roc_curve(DataLabel, np.array(pre_sc_poly).flatten(), pos_label=1)
    fpr_sigmoid, tpr_sigmoid, thresholds_sigmoid = roc_curve(DataLabel, np.array(pre_sc_sigmoid).flatten(), pos_label=1)
    #画图
    plt.figure("SVC_ALL")

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.plot(linear_fpr, linear_tpr, '--', label='linear_svc ROC (area = %0.2f,' % auc(linear_fpr, linear_tpr)+'accuracy = %0.2f)' % linear_Accuracy, lw=2)
    plt.plot(fpr_linear, tpr_linear, '--', label='svc_linear (area = %0.2f,' % auc(fpr_linear, tpr_linear)+'accuracy = %0.2f)' % Accuracy_linear, lw=2)
    plt.plot(fpr_rbf, tpr_rbf, '--', label='svc_rbf (area = %0.2f,' % auc(fpr_rbf, tpr_rbf)+'accuracy = %0.2f)' % Accuracy_rbf, lw=2)
    plt.plot(fpr_poly, tpr_poly, '--', label='svc_poly (area = %0.2f,' % auc(fpr_poly, tpr_poly)+'accuracy = %0.2f)' % Accuracy_poly, lw=2)
    plt.plot(fpr_sigmoid, tpr_sigmoid, '--', label='svc_sigmoid (area = %0.2f,' % auc(fpr_sigmoid, tpr_sigmoid)+'accuracy = %0.2f)' % Accuracy_sigmoid, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC_ALL ROC')
    plt.legend(loc="lower right")
    plt.savefig("SVC_ALL ROC")
    plt.show()

#非线性SVC 核函数为sigmoid 测试gamma coef0
def SVC_sigmoid(DataBJ76_pca,DataLabel,loo):
    plt.figure("svc_sigmoid",figsize=(25,5))
    #测试gamma
    plt.subplot(1,2,1)
    GG=range(1,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for gamma in GG:
        pre_sc=[]
        svc_sigmoid = sklearn.svm.SVC(kernel='sigmoid',gamma=gamma,coef0=0)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_sigmoid, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_sigmoid.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for G in GG:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(GG, AA, '--', label='accuracy', lw=2)
    plt.plot(GG, linear_auc, '--', label='auc', lw=2)
    plt.plot(GG, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(GG, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of gamma')
    plt.ylabel('accuracy')
    plt.title('SVC_sigmoid_gamma')
    plt.legend(loc="lower right")


    #测试Coef0
    plt.subplot(1,3,3)
    CC=range(0,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for C in CC:
        pre_sc=[]
        svc_sigmoid = sklearn.svm.SVC(kernel='sigmoid',gamma=0.010,coef0=C)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_sigmoid, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_sigmoid.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for C in CC:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(CC, AA, '--', label='accuracy', lw=2)
    plt.plot(CC, linear_auc, '--', label='auc', lw=2)
    plt.plot(CC, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(CC, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of coef0')
    plt.ylabel('accuracy')
    plt.title('SVC_sigmoid_coef0 ')
    plt.legend(loc="lower right")

    plt.savefig("SVC_sigmoid")
    plt.show()

#非线性SVC 核函数为rbf 测试gamma
def SVC_rbf(DataBJ76_pca,DataLabel,loo):
    plt.figure("svc_rbf",figsize=(25,5))
    #测试gamma
    GG=range(1,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for gamma in GG:
        pre_sc=[]
        svc_rbf = sklearn.svm.SVC(kernel='rbf',gamma=gamma)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_rbf, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_rbf.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for G in GG:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(GG, AA, '--', label='accuracy', lw=2)
    plt.plot(GG, linear_auc, '--', label='auc', lw=2)
    plt.plot(GG, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(GG, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of gamma')
    plt.ylabel('accuracy')
    plt.title('SVC_rbf_gamma')
    plt.legend(loc="lower right")

    plt.savefig("SVC_rbf")
    plt.show()

#非线性SVC 核函数为poly 测试degree gamma coef0
def SVC_poly(DataBJ76_pca,DataLabel,loo):
    plt.figure("svc_poly",figsize=(25,5))
    #测试degree
    plt.subplot(1,3,1)
    DD=range(1,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for degree in DD:
        pre_sc=[]
        svc_poly = sklearn.svm.SVC(kernel='poly',degree=degree)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_poly, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_poly.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for D in DD:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(DD, AA, '--', label='accuracy', lw=2)
    plt.plot(DD, linear_auc, '--', label='auc', lw=2)
    plt.plot(DD, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(DD, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of degree')
    plt.ylabel('accuracy')
    plt.title('SVC_poly_degree')
    plt.legend(loc="lower right")

    #测试gamma
    plt.subplot(1,3,2)
    GG=range(1,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for gamma in GG:
        pre_sc=[]
        svc_poly = sklearn.svm.SVC(kernel='poly',gamma=gamma,degree=3)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_poly, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_poly.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for G in GG:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(GG, AA, '--', label='accuracy', lw=2)
    plt.plot(GG, linear_auc, '--', label='auc', lw=2)
    plt.plot(GG, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(GG, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of gamma')
    plt.ylabel('accuracy')
    plt.title('SVC_poly_gamma')
    plt.legend(loc="lower right")


    #测试Coef0
    plt.subplot(1,3,3)
    CC=range(0,20)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for C in CC:
        pre_sc=[]
        svc_poly = sklearn.svm.SVC(kernel='poly',gamma=10,degree=3,coef0=C)
        # 计算预测标签
        pre_Label = cross_val_predict(svc_poly, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = svc_poly.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for C in CC:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(CC, AA, '--', label='accuracy', lw=2)
    plt.plot(CC, linear_auc, '--', label='auc', lw=2)
    plt.plot(CC, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(CC, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of coef0')
    plt.ylabel('accuracy')
    plt.title('SVC_poly_coef0 ')
    plt.legend(loc="lower right")

    plt.savefig("SVC_poly")
    plt.show()

#SVC 非线性核函数为 linear
def SVC_Linear(DataBJ76_pca,DataLabel,loo):
    pre_sc = []
    random_state = np.random.RandomState(0)
    svc_linear = sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state)
    # 计算预测标签
    pre_Label = cross_val_predict(svc_linear,DataBJ76_pca,DataLabel,cv=loo)
    #计算正确率
    Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True]/76
    #循环迭代
    for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
        probas_ = svc_linear.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
        pre_sc.append(probas_)
    fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
    #画图
    plt.figure("SVC_Linear")
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '--', label='ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC_Linear ROC (accuracy = %0.2f)' % Accuracy)
    plt.legend(loc="lower right")
    plt.savefig("SVC_Linear ROC")
    plt.show()

#SVC 线性SVM 测试loss
def LinearSVC(DataBJ76_pca,DataLabel,loo):
    losses=['hinge','squared_hinge']
    plt.figure("LinearSVC",figsize=(25,5))
    #测试loss
    plt.subplot(1,2,1)
    for loss in losses:
        pre_sc = []
        linear_svc = sklearn.svm.LinearSVC(loss=loss)
        # 计算预测标签
        pre_Label = cross_val_predict(linear_svc,DataBJ76_pca,DataLabel,cv=loo)
        #计算正确率
        Accuracy = Counter(list(pre_Label == DataLabel.flatten()))[True]/76
        #循环迭代
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = linear_svc.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        mean_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, '--', label=loss+':ROC (auc = %0.2f'% mean_auc +',accuracy = %0.2f )'% Accuracy , lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('linearSVC_loss ROC ')
    plt.legend(loc="lower right")

    #测试C
    plt.subplot(1,2,2)
    CC=np.logspace(-2,1)
    AA=[]
    mean_AA=[]
    linear_auc=[]
    mean_linearauc=[]
    for C in CC:
        pre_sc=[]
        linear_svc = sklearn.svm.LinearSVC(C=C)
        # 计算预测标签
        pre_Label = cross_val_predict(linear_svc, DataBJ76_pca, DataLabel, cv=loo)
        # 计算正确率
        Accuracy = Counter(list(pre_Label  == DataLabel.flatten()))[True] / 76
        AA.append(Accuracy)
        for train_index, test_index in loo.split(DataBJ76_pca, DataLabel):
            probas_ = linear_svc.fit(DataBJ76_pca[train_index], DataLabel[train_index]).decision_function(DataBJ76_pca[test_index])
            pre_sc.append(probas_)
        fpr, tpr, thresholds = roc_curve(DataLabel, np.array(pre_sc).flatten(), pos_label=1)
        linear_auc.append(auc(fpr, tpr))
    for C in CC:
        mean_AA.append(np.mean(AA))
        mean_linearauc.append(np.mean(linear_auc))
    plt.plot(CC, AA, '--', label='accuracy', lw=2)
    plt.plot(CC, linear_auc, '--', label='auc', lw=2)
    plt.plot(CC, mean_AA, 'k--', color=(0.6, 0.6, 0.6), label='mean accuracy %0.2f' %np.mean(AA))
    plt.plot(CC, mean_linearauc, 'k--', color=(0.3, 0.3, 0.4), label='mean auc %0.2f' % np.mean(linear_auc))
    plt.xlabel('the range of C')
    plt.ylabel('accuracy')
    plt.title('linearSVC_C ROC ')
    plt.legend(loc="lower right")

    plt.savefig("LinearSVC ROC")
    plt.show()


if __name__=='__main__':
    #加载数据集
    AALCorrArray_BJ_63 = sio.loadmat('AALCorrArray_BJ_63.mat')
    Age_BJ_63 = sio.loadmat('Age_BJ_63.mat')
    #AALCorrArray_BJ_76_Class = sio.loadmat('AALCorrArray_BJ_76_Class.mat')
    Age_BJ_76_Class = sio.loadmat('Age_BJ_76_Class.mat')
    #回归数据集
    DataAALCorrArray_BJ_63 = AALCorrArray_BJ_63['Data']
    #SVR降维处理
    DataBJ_63_pca = PCA_com(DataAALCorrArray_BJ_63)
    DataAge_BJ_63 = Age_BJ_63['Label']
    #SVR 分析
    SVR_Anylise(DataBJ_63_pca, DataAge_BJ_63)
    #分类数据集
    DataBJ76 = Age_BJ_76_Class['Data']
    #SVC降维
    DataBJ76_pca = PCA_com(DataBJ76)
    DataLabel = Age_BJ_76_Class['Label']
    #SVC分析
    SVC_Anylise(DataBJ76_pca, DataLabel)

