'''
只跑一遍
'''
import osmnx as ox
import os

G = ox.graph_from_place('Beijing, China',network_type = 'drive',which_result=2)
ox.save_graphml(G, filename='BeijingStreetMap') #将数据储存到文件BeijingStreetMap中

import osmnx as ox
import os
import pandas as pd
import networkx as nx
import numpy as np
import math

def getAccidentData(fileName):
    col_names = ["Time","address","lat","lang"]
    accident = pd.read_csv(fileName,names = col_names,encoding ="GB2312")
    accident["Time"] = getTimeSegment(accident["Time"])
    return accident

def getTimeSegment(TimeString):
    base = pd.to_datetime("00:00")
    return [math.floor(pd.Timedelta(i-base).seconds/(60*60)) for i in pd.to_datetime(TimeString)]


def getAllNeighbor(nodeList,G):
    neighbor_nodes=[]
    for i in nodeList:
        if i in G.nodes():
            neighbor_id = nx.neighbors(G,i)
            for j in neighbor_id:
                neighbor_nodes.append(j)
            neighbor_nodes.append(i)
    return neighbor_nodes


def getAllTimeNeighbor(predict,G):
	'''
	得到所有路段的时间点
	G为地图
	'''
	neighbor={}
    for i in range(predict.shape[0]):
        neighbor[i]=getAllNeighbor([predict["id1"][i],predict["id2"][i]],G)
    return neighbor

'''
根据accident数据找到所有最近点
'''
def getAllNearestId(accident):
    array = np.zeros((accident.shape[0],2))
    Time_id = pd.DataFrame(array,columns = ["time","id"])
    for i in range(accident.shape[0]):
        Time_id["id"][i] = ox.get_nearest_node(G,(accident["lat"][i],accident["lang"][i]))
        Time_id["time"][i] = accident["Time"][i]
    return Time_id
        

'''
neighbor为每一条预测结果的邻近点
timeSegment为预测数据的时间点，和上一个矩阵顺序相同
Time_id为真实异常点的发生时间段和最近id
'''
def countPridectRight(neighbor,Time_id,timeSegment):
    TP=0
    for i in range(Time_id.shape[0]):
        index = timeSegment[timeSegment == Time_id["time"][i]].index
        print(len(index))
        if len(index)!=0:
            print(index)
            for j in index:
                print(neighbor[j])
                print(Time_id["id"][i])
                if Time_id["id"][i] in neighbor[j]:
                     TP+=1
    return TP
    
def getFinalMartrix(test,data,threshold):   
    p = np.where(data.values<threshold)
    array = np.zeros((p[0].size,3))
    Matric = pd.DataFrame(array, columns = ["id1","id2","time"])
    for i in range(p[0].size):
        Matric["id1"][i] = test[0][p[0][i]]
        Matric["id2"][i] = test[1][p[0][i]]
        Matric["time"][i] = p[1][i]-1
    return Matric
	

G = ox.load_graphml('BeijingStreetMap')
test = pd.read_csv(".\\test\\Flow20111129.csv",header=None)	#导入测试矩阵
data = pd.read_csv(".\\result.csv")		#导入概率矩阵
accident = getAccidentData(".\\test\\Accident20111129.csv")
Time_id = getAllNearestId(accident) #找到事故发生点的点的ID
predict=getFinalMartrix(test,data,0.025) #通过概率函数获得预测矩阵
timeSegment = predict["time"]
neighbor = getAllTimeNeighbor(predict,G) 	#找到每一条预测数据的所有的临近点
rightNumber = countPridectRight(neighbor,Time_id,timeSegment) #预测准确的数据
print(rightNumber)
p = rightNumber/predict.shape[0]
r = rightNumber/accident.shape[0]
print(p)
print(r)

