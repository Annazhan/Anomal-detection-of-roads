import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm
import pandas as pd
import os
import nimfa
from math import floor
from sklearn.cluster import KMeans
import osmnx as ox
import folium
import math

tooLittleStd = 5e-5 #最小能接受的标准差
trueDistance = 3000 #判断在事故点偏移的标准
earth_radius = 6378.137

def draw_pic(n_clusters, labels, X):
    ''' 开始绘制图片 '''
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=3)


    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()

"""
data是已经选出来在某一类中需要计算的值
"""
def probability(mean,std,data):
    return norm(mean,std).pdf(data)


def NMF_forCoef(path):
    """
    path为存放流量矩阵的位置(文件夹)
    NMF分解
    """
    fileName = os.listdir(path)
    V = np.zeros((127049,96))

    for i in range(len(fileName)):
        filePath = path+"\\"+fileName[i]
        data = pd.read_csv(filePath,header=None)
        data.drop(axis = 1,columns = [0,1,2],inplace = True)
        V = V+data.values
    
    nmf = nimfa.Nmf(V/9, max_iter=200, rank=3, update='euclidean', objective='fro')
    nmf_fit = nmf()
    return np.array(nmf_fit.basis()) #系数矩阵(127049x3)


def cluster(data,clusterNumber):
    """
    KMeans聚类
    """
    cluster = KMeans(init='k-means++', n_clusters=clusterNumber, n_init=10).fit(data)
    #print("cluster is over")
    return cluster.labels_


def isMakeSense(mean,std):
    """
    如果平均值和方差都接近零，则正态分布没有意义
    paramter : 
    ------------
    mean : float
        一个向量的平均数
    std : float
        一个向量的标准差

    return
    -----------------
    -1 : int
        道路一定正常，将其对应的某时刻道路设置为0.5
    0 : int
        标准差非常小，重新设置标准差
    1 : int
        正常计算
    """
    if mean == 0:
        return -1   #道路一定正常
    elif std <= tooLittleStd:
        return 0    #标准差较小
    else: 
        return 1    #正常情况


def adjcentMatrix(predictMatrix,labels,clusterNumber):
    """
    计算临界概率矩阵
    paramters
    ---------------
    predictMatrix : numpy.array
        需要预测的矩阵（已经处理好）
    labels : np.array
        对每一列的标签
    clusterNumber : int
        实际分类的类别
    
    return
    --------------------
    adjcentProb : np.array
        邻居模型返回的概率矩阵
    """
    segNum = predictMatrix.shape[1]
    adjcentProb = np.zeros(predictMatrix.shape)
    for segTime in range(segNum):
        for index in range(clusterNumber):
            col = predictMatrix[:,segTime][labels==index]
            isSensible = isMakeSense(col.mean(),col.std())
            if isSensible == 1:
                #print("neccesary:",index,", cluster number:",len(labels[labels==index]))
                #print("mean:",col.mean())
                #print("std:",col.std())
                #print(col)
                #os.system("pause")
                #print(np.array(probability(col.mean(),col.std(),col)))
                adjcentProb[:,segTime][labels==index] = np.array(probability(col.mean(),col.std(),col))
            else:
                #print("unneccesary:",index,", cluster number:",len(labels[labels==index]))
                adjcentProb[:,segTime][labels==index] = 0.5        
    return adjcentProb



def changeTimeSeg(data):
    """
    将15分钟合成1个小时

    paramter
    ------------
    data : np.array
        去掉前面对模型建立无用的3列

    return
    -----------
    newMatrix : np.array
        返回将15分钟合并成1小时的矩阵
    """
    newMatrix = np.zeros((data.shape[0],floor(data.shape[1]/4)))
    for i in range(floor(data.shape[1]/4)):
        newMatrix[:,i] = data[:,4*i:4*i+4].sum(axis=1)
    return newMatrix


def preProcessing(path):
    """
    对读入的矩阵预处理
    
    paramter
    ----------------
    path : str
        需要处理的流量矩阵在磁盘中的位置
    
    return
    -------------------
    np.array
        预处理完成的矩阵
    """
    data = pd.read_csv(path,header=None)
    data.drop(axis = 1,columns = [0,1,2],inplace = True)
    return changeTimeSeg(data.values)


def concatMatrix(dir):
    """
    将9个矩阵合并

    paramter
    -----------
    dir : str
        流量矩阵存放的位置

    return
    ---------------
    tmp1: np.array
        将流量矩阵在列方向上合并之后的矩阵
    """
    fileName = os.listdir(dir)
    tmp1 = preProcessing(dir+"\\"+fileName[0])
    for i in range(1,len(fileName)):
        tmp2 = preProcessing(dir+"\\"+fileName[i])
        tmp1 = np.concatenate((tmp1,tmp2),axis=1)
    return tmp1



def concatSet(data,index,time,matrixNum,segNum):
    """
    将9列合并

    paramters
    ------------
    data : np.array
        从中选取数据的矩阵
    index : np.array
        有lebel得知的所需要的行数
    time : int
        某一时段
    matrixNum : int
        需要合并的个数
    segNum : int
        需要跳过的列数

    return
    ---------------
    tmp1 : np.array
        合并之后的向量

    """
    tmp1 = data[index,:][:,time]
    for n in range(1,matrixNum):
        tmp2 = data[index,:][:,time+n*segNum]
        tmp1 = np.concatenate((tmp1,tmp2))
    return tmp1


def historyMatrix(dir,labels,predictData,clusterNumber):
    """
    选取历史信息，一天24小时
    
    paramters
    ---------
    dir : str
        放置道路车流量的文件夹.
    labels : np.array
        根据聚类分类之后每一列的标签
    predictData : np.array
        需要预测的道路车流量矩阵
    clusterNumber : int
        实际分类的类别数

    return
    -----------
    historyProb : np.array
        返回历史模型的概率矩阵

    """
    data = concatMatrix(dir)
    matrixNum = len(os.listdir(dir))
    historyProb = np.zeros(predictData.shape)
    for time in range(predictData.shape[1]):
        for i in range(clusterNumber): #i为聚类的类别
            index = np.argwhere(labels==i)[:,0] #index为类别找到的行数
            col = concatSet(data,index,time,matrixNum,predictData.shape[1])
            if isMakeSense(col.mean(),col.std()) == -1:
                #print("totally, means:",col.mean(),", std:",col.std(),", size:",len(col))
                historyProb[:,time][labels==i] = 0.5
            elif isMakeSense(col.mean(),col.std()) == 0:
                #print("partitial, means:",col.mean(),", std:",col.std(),", size:",len(col))
                historyProb[:,time][labels == i] = np.array(probability(col.mean(),tooLittleStd, predictData[:,time][labels==i]))
            else:
                #print("make sense, means:",col.mean(),", std:",col.std(),", size:",len(col))
                historyProb[:,time][labels == i] = np.array(probability(col.mean(),col.std(), predictData[:,time][labels==i]))
    return historyProb


def judgeModel(probability, threshold, authenticPath, predictPath,saveDir=None):
    """
    对模型计算查准率和查全率，进而对模型进行评价

    parameter
    ----------------
    probability : np.array
        最后算出的概率矩阵
    threshold : float
        异常阈值
    authenticPath : str
        存放真实异常数据的位置
    predictPath : str
        存放流量数据的位置
    saveDir : str
        存放地图的位置

    return
    --------------------
    precise : float
        查准率
    recall : float
        查全率
    """
    index = np.argwhere(probability<threshold)
    roadId = (pd.read_csv(predictPath,header=None,usecols=[0,1])).values
    predictAccident = np.concatenate((index[:,1].reshape(index.shape[0],1),roadId[index[:,0]]),axis=1) #最终的预测事故点矩阵
    accident = readAccidentFile(authenticPath) #真实事故点矩阵
    nodes = (ox.load_graphml('BeijingStreetMap')).nodes(data=True) #地图上的点
    realTP, prdictTP = 0,0
    flag = False
    for i in range(accident.shape[0]):
        nowPredict = predictAccident[predictAccident[:,0]==accident[i,0]]
        for j in range(nowPredict.shape[0]):
            if isRightAccident(accident[i,2],accident[i,3],nowPredict[j,1],nowPredict[j,2],nodes):
                flag = True
                prdictTP+=1
        if flag:
            realTP+=1
            flag = False
            if saveDir !=None:
                drawMap(accident[i,:],nowPredict,saveDir,nodes)            

    return prdictTP/(predictAccident.shape[0]), realTP/(accident.shape[0])


def drawMap(accident,nowPredict,saveDir,nodes):
    """
    画出地图
    """
    m = folium.Map(location=[39.9042, 116.4074], tiles="OpenStreetMap", zoom_start=10)
    tooltip = "click me! 0.0"
    folium.Marker(location = [accident[2],accident[3]], popup='<b>True accident!!!</b>', tooltip=tooltip,icon=folium.Icon(color='red')).add_to(m) 
    for j in range(nowPredict.shape[0]):
        if nowPredict[j,1] in nodes:
            n = [nodes[nowPredict[j,1]]['y'],nodes[nowPredict[j,1]]['x']]
            folium.Marker(location = n, popup='<b>predict accident!!!</b>', tooltip=tooltip,icon=folium.Icon(color='blue')).add_to(m)
        if nowPredict[j,2] in nodes:
            n = [nodes[nowPredict[j,2]]['y'],nodes[nowPredict[j,2]]['x']]
            folium.Marker(location = n, popup='<b>predict accident!!!</b>', tooltip=tooltip,icon=folium.Icon(color='blue')).add_to(m)
    filePath = saveDir+"\\"+str(accident[0])+"segment"+str(accident[1])+".html"
    print(filePath)
    m.save(filePath)

def readAccidentFile(path):
    """
    读取真实事故数据文件
    """
    col_names = ["Time","address","lat","lang"]
    accident = pd.read_csv(path,names = col_names,encoding ="GB2312")
    accident["Time"] = getTimeSegment(accident["Time"])
    return accident.values

def getTimeSegment(TimeString):
    """
    处理真实事故数据文件
    """
    base = pd.to_datetime("00:00")
    return [math.floor(pd.Timedelta(i-base).seconds/(60*60)) for i in pd.to_datetime(TimeString)]


def isRightAccident(accLat,accLang,id1,id2,nodes):
    """
    判断是否是真实事故点
    """
    isTrue = False
    if id1 in nodes:
        if calDistance(accLat,accLang,nodes[id1]['y'],nodes[id1]['x']) < trueDistance:
            isTrue = True
    elif id2 in nodes:
        if calDistance(accLat,accLang,nodes[id2]['y'],nodes[id2]['x']) < trueDistance:
            isTrue = True
    return isTrue

def calDistance(lat1, lang1, lat2, lang2):
    """
    由经纬度计算距离
    """
    latRad1 = math.radians(lat1)
    langRad1 = math.radians(lang1)
    latRad2 = math.radians(lat2)
    langRad2 = math.radians(lang2)
    
    a = latRad1-latRad2
    b = langRad1-langRad2
    s = 2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+math.cos(latRad1)*math.cos(latRad2)*math.pow(math.sin(b/2),2))) 
    s = s*earth_radius*1000
    return s