from utils import draw_pic
from utils import NMF_forCoef
from utils import cluster
from utils import adjcentMatrix
from utils import historyMatrix
from utils import preProcessing
from utils import judgeModel
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

path = "D:\\jupyter_notebook\\北京寒假冬令营\\16_28workDay" #存放流量矩阵的文件夹
predictPath = "D:\\jupyter_notebook\\北京寒假冬令营\\test\\Flow20111129.csv" #预测矩阵
saveDir = ".\\result" # 存放矩阵的位置
authenticPath = "D:\\jupyter_notebook\\北京寒假冬令营\\test\\Accident20111129.csv"  #真实异常数据
predictData = preProcessing(predictPath)

#######################################################################################
beta = 0.8 #设置两个模型占比
threshold = 1e-75 # 预测异常的阈值

###########################################################################################
# 非负矩阵分解,H为系数矩阵
W = NMF_forCoef(path)
print("fractorization is over and W's shape is",W.shape)
#################################################################################################
#聚类
clusterNumber = 200 #聚类个数
labels = cluster(W,clusterNumber)
del W
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # 类的真实数目
print("actually cluster number is ",n_clusters)
#draw_pic(n_clusters, labels, W) #画出聚类图

####################################################################################################
#根据label建立历史模型
historyPredict = historyMatrix(path,labels,predictData,n_clusters) 
np.savetxt(saveDir+"\\historyProb.csv", historyPredict, delimiter = ',')

######################################################################################################
#根据label建立邻居模型
adjcentPredict = adjcentMatrix(predictData,labels,n_clusters)
np.savetxt(saveDir+"\\adjcentProb.csv", adjcentPredict, delimiter = ',')

########################################################################################################
probability = (1-beta)*adjcentPredict+beta*historyPredict
del adjcentMatrix,historyPredict
precise,recall = judgeModel(probability,threshold, authenticPath, predictPath, saveDir="..\\result")
print("precise: ",precise,", recall: ",recall)
