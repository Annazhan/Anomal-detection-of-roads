import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from boto import sns
from sklearn.cluster import KMeans
from scipy.optimize import leastsq
from scipy import stats
from scipy.stats import norm

filePath = "D://Data Mining/寒假营/数据/"

def runFile(name):
    data = pd.read_csv(filePath + name, header=None, encoding="gbk")
    data = data.iloc[0:, 3:]
    data.columns = data.columns - 3
    data = data.values
    return data  # 返回矩阵

################################################################
from sklearn.decomposition import NMF

def NMF_decomposition(data, n_components):
    model = NMF(n_components=n_components, random_state=0, init='nndsvd')
    Matrix_M = model.fit_transform(data)
    Matrix_H = model.components_
    return Matrix_M, Matrix_H


# fileList = os.listdir(filePath)
# account=0
# for each in fileList:
#     account+=1
#     data=pd.read_csv("D://Data Mining/寒假营/数据/"+each,header=None)
#     data = data.iloc[0:, 3:]
#     data.columns = data.columns - 3
#     data_matrix = data.values
#     Matrix_M, Matrix_H = NMF_decomposition(data,3)
#     M_D = pd.DataFrame(Matrix_M)
#     H_D = pd.DataFrame(Matrix_H)
#     M_D.to_csv("D://Experiment/M_Matrix"+str(account)+".csv",header=None)
#     H_D.to_csv("D://Experiment2/H_Matrix"+str(account)+".csv",header=None)

###############################################################
def kMeans(clusters_num, data):
    prediction = KMeans(n_clusters=clusters_num).fit(data)
    return prediction


################################################################
# def buildHistoricalModel(fileList):
# account=0
# for each in fileList:
#  account+=1
#  data=pd.read_csv("D://Data Mining/寒假营/数据/"+each)
#  data=data.iloc[0:3,:]
#  data.columns=data.columns-3
#  # print("第",account,"次，data形状为",data.shape)
#  data=data.values
#  if(account==0):
#      Mat_Avg=np.zeros(shape=(data.shape[0],data.shape[1]))
#      Mat_Std=np.zeros(shape=(data.shape[0],data.shape[1]))
#  Matrix_M,Matrix_H=NMF_decomposition(data)
#  Mat_Avg=Mat_Avg+Matrix_M
#  Mat_Std=Mat_Std+Matrix_M
# row_num = Matrix_M.shape[0]
# column_num = Matrix_H.shape[1]
# record=[]
# for i in row_num:
#     for j in column_num:
#         record[]

# def create_Mean_Std_Matix(fileList):
#     fileLength = len(fileList)  # 文件个数
#     print("文件长度为：", fileLength)
#
#     data_matrix = []
#     for each in fileList:
#         data = pd.read_csv("D://Experiment/" + each, header=None)
#         data = data.iloc[:, 1:]
#         data.columns = data.columns - 1
#         data_matrix.append(data.values)  # 得到该文件的数据矩阵
#
#     row_num = data_matrix[0].shape[0]
#     col_num = data_matrix[0].shape[1]
#     mean_matrix = np.random.random((row_num, col_num))
#     std_matrix = np.random.random((row_num, col_num))
#     print("这些数据每天的矩阵的行列数为:", row_num, " ", col_num)
#     for i in range(row_num):
#         for j in range(col_num):
#             tem_list = []
#             for index in range(len(data_matrix)):
#                 tem_list.append(data_matrix[index][i, j])
#             # 添加完成之后进入计算平均数和标准差
#             average = np.average(tem_list)
#             std = np.std(tem_list)
#             mean_matrix[i, j] = average
#             std_matrix[i, j] = std
#     mean_matrix = pd.DataFrame(mean_matrix)
#     std_matrix = pd.DataFrame(std_matrix)
#     # 将Mean和Std矩阵写入表格
#     mean_matrix.to_csv("D://Experiment3/Mean_Matrix.csv", header=None)
#     std_matrix.to_csv("D://Experiment3/Std_Matrix.csv", header=None)


# 返回正态分布的范式
# def calProDen(average, sigma):
#     return scipy.stats.norm(average, sigma)


# def loadModel():
#     fileList = os.listdir("D://Experiment/")  # 读M矩阵
#     create_Mean_Std_Matix(fileList)  # 写入Mean_Matrix Std_Matrix
#     mean_matrix = pd.read_csv("D://Experiment3/Mean_Matrix.csv", header=None)
#     std_matrix = pd.read_csv("D://Experiment3/Std_Matrix.csv", header=None)
#     mean_matrix = mean_matrix.iloc[:, 1:]
#     mean_matrix.columns = mean_matrix.columns - 1
#     mean_matrix = mean_matrix.values
#     std_matrix = std_matrix.iloc[:, 1:]
#     std_matrix.columns = std_matrix.columns - 1
#     std_matrix = std_matrix.values
#     rol_num = mean_matrix.shape[0]
#     col_num = mean_matrix.shape[1]
#     print("rol_num 和col分别为:", rol_num, " ", col_num)
#
#     # 记载检验数据集
#     test_matrix = pd.read_csv("D://Experiment4/20111115.csv", header=None)
#     test_matrix.drop([0, 1, 2], axis=1, inplace=True)
#     test_matrix.columns = test_matrix.columns - 3
#     test_matrix = test_matrix.values
#     test_matrix_M, test_matrix_H = NMF_decomposition(test_matrix, 3)
#
#     probability = []
#     for i in range(rol_num):
#         for j in range(col_num):
#             formula = calProDen(mean_matrix[i, j], std_matrix[i, j])
#             probability.append(formula.pdf(test_matrix_M[i, j]))
#     plt.plot(probability, label='Probability')
#     plt.legend()
#     plt.show()
#     return probability


# probability=loadModel()
##########################################################################
# Example below:
# M = pd.read_csv("D://Experiment/M_Matrix1.csv", header=None)
# M = M.iloc[:, 1:]
# M=M.values
# print(M)
# prediction = kMeans(20, M)
# label = pd.DataFrame(prediction.labels_)
# label=label.values.T
# lab=label.tolist()[0]
###########################################################################
def concatColumns(data):
    times=int(data.shape[1]/4)
    #两列两列合并
    account=0
    for i in range(times):
        data[account]=data[account]+data[account+1]+data[account+2]+data[account+3]
        data.drop([account+1,account+2,account+3],axis=1,inplace=True)
        account+=4
    return data

def getData(file):    #输入文件路径得到数据数列
    data=pd.read_csv(file,header=None)
    return data.values
def getMod(vector):  #得到模型公式
    std1=np.std(vector)
    if(std1==0.0):
        std1+=0.000001
    return norm(np.mean(vector),std1)

def getPro(model,a):    #得到概率
    return round(model.cdf(a),6)

def findFittableIndex(labels,n):  #查找符合object的所有索引值
    list1=[]
    for each in range(n):
        list1.append([])
        list1[each]=[i for i,a in enumerate(labels) if a==each]
    return list1

def collectAllClusters(list1,n):
    cur_lst=[]
    for i in range(n):
        index=findFittableIndex(list1,i)   #查找这个类具体有哪些元素，返回列表
        cur_lst.append(index)
    cluster_dic={}
    for i in range(n):
        cluster_dic[i]=cur_lst[i]
    return cluster_dic     #每一类中具体有哪些，用字典表示返回

def getSamples(data,when,cluster_index):    #
    result=[]
    for i in range(cluster_index):
        tem_var=[]
        index = indexset[i]
        for j in index:
            tem_var.append(data[j][when])   #这个时段属于哪个类，从而进入这个类的列表
        result.append(tem_var)
    return np.array(result)      #返回列表去获取norm

def train_model(data,cluster_index):
    list1=[]
    for i in range(cluster_index):   #每个类依次
        list2=[]
        for j in range(data.shape[1]):   #按照每个时段来，一列一列遍历
            list2.append(getMod( getSamples(data,j,cluster_index)[i] ))
        list1.append(list2)
    return list1
###############################################################
# data=pd.read_csv("D://Experiment4/20111115.csv",header=None)
# data.drop([0,1,2],axis=1,inplace=True)
# data=data.values
# M_Matrix=NMF_decomposition(data,3)
# prediction=kMeans(20,M_Matrix)
# labels=prediction.labels_      #得到该数据的具体分类
# print("得到了20111115.csv的聚类类别为：",labels)

data_matrix=pd.read_csv("D://Data Mining/寒假营/数据/20111116.csv",header=None)
data_matrix=data_matrix.iloc[:,3:]
# data_matrix.columns=data_matrix.columns-3
# data_matrix=concatColumns(data_matrix)
data_matrix=data_matrix.values
fileList=os.listdir(filePath)
fileList.pop(0)
for each in fileList:
    data_matrix += runFile(each)
data_matrix = data_matrix / 9  # 九天的流量平均数
data_matrix_M,data_matrix_H=NMF_decomposition(data_matrix,3)
pd.DataFrame(data_matrix_M).to_csv("D://Experiment/data_matrix_M.csv",header=None)
pd.DataFrame(data_matrix_H).to_csv("D://Experiment/data_matrix_H.csv",header=None)
prediction=kMeans(250,data_matrix_M)
labels=prediction.labels_
print("得到了20111115.csv的聚类类别为：",labels)

indexset=findFittableIndex(labels,250)
print("得到了每个类中具体包含了哪些值",indexset)

#开始读取数据
target=pd.read_csv("D://Experiment4/20111128.csv",header=None)
target=target.iloc[:,3:]
target.columns=target.columns-3
target=concatColumns(target)
target=target.values
storage_probability=[]    #存放概率的列表
finalmodel=train_model(target,250)
for i in range(target.shape[0]):
    cluster=labels[i]
    tem_lst=[]
    for j in range(target.shape[1]):
        tem_lst.append(getPro(finalmodel[cluster][j],target[i][j]))
        print(str(i)+" "+str(j)+"结束！")
    storage_probability.append(tem_lst)

df=pd.DataFrame(storage_probability)
df.to_csv("D://Experiment2/邻居道路得分.csv",header=None)
##############################################################

#historical score
#先将聚类分好，每个类别计算Mean和Std

#labels



def searchAllMatrix():
    matrix_list=[]
    data_matrix = pd.read_csv("D://Data Mining/寒假营/数据/20111116.csv", header=None)
    data_matrix = data_matrix.iloc[:, 3:]
    data_matrix.columns=data_matrix.columns-3
    data_matrix=concatColumns(data_matrix)
    data_matrix = data_matrix.values
    matrix_list.append(data_matrix)

    filelist = os.listdir(filePath)
    filelist.pop(0)
    for each in filelist:
        data_matrix=pd.read_csv("D://Data Mining/寒假营/数据/"+each,header=None)
        data_matrix=data_matrix.iloc[:,3:]
        data_matrix=data_matrix.values
        matrix_list.append(data_matrix)
    return matrix_list

def building_model(data):
    matrix_list=searchAllMatrix()   #matrix_list存储了九天的数据

    rol_num=data.shape[0]
    col_num=data.shape[1]
    print("行：",rol_num," 列：",col_num)
    storage_probability_history=np.zeros(shape=(rol_num,col_num))
    for i in range(rol_num):
        # this_type = labels[i]  # 这条道路所属的类别
        # print("这条道路所属类别为:","this_type类")
        # indexs = indexset[this_type]
        # indexs.remove(i)
        # indexs_length=len(indexs)
        # print("这条道路的邻居有：",indexs)
        for j in range(col_num):
            cur_list = []
            for each in matrix_list:
                cur_list.append(each[i,j])  #存放这条道路这个时段
                # if(indexs_length<20):
                #       for every in indexs:      #这一天的所有邻居也放进去
                #           cur_list.append(each[every,j])
                # else:
                #    count=0
                #    while(count<20):
                #          cur_list.append( each[indexs[count-1],j] )
                #          count+=1
            formula=getMod(cur_list)   #得到这条道路这个时段的正态分布
            probability=getPro(formula,data[i,j])
            print("这个cur_list里的元素有",len(cur_list),"个,算出来的道路: ",i," 时段：",j," 的概率为：",probability)
            storage_probability_history[i,j]=probability
    return storage_probability_history


storage_probability2=building_model(target)
df2=pd.DataFrame(storage_probability2)
df2.to_csv("D://Experiment2/历史道路得分.csv",header=None)

def getScoreMatrix(matrix1,matrix2,beita):  #beita=β
    score_matrix=beita*matrix1+(1-beita)*matrix2
    return score_matrix

result1=np.array(storage_probability)
result2=np.array(storage_probability2)
score_matrix=getScoreMatrix(result1,result2,1)
df3=pd.DataFrame(score_matrix)
df3.to_csv("D://Experiment3/总得分.csv",header=None)


