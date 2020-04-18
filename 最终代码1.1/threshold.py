import pandas as pd
import numpy as np
from utils import judgeModel

np.set_printoptions(threshold=np.inf)

beta = 0.8
threshold=1e-75
adjcentProb = (pd.read_csv(".\\result\\adjcentProb.csv",header=None)).values
historyProb = (pd.read_csv(".\\result\\historyProb.csv",header=None)).values
probability = (1-beta)*adjcentProb+beta*historyProb
del adjcentProb,historyProb

predictPath = "D:\\jupyter_notebook\\北京寒假冬令营\\test\\Flow20111129.csv"
authenticPath = "D:\\jupyter_notebook\\北京寒假冬令营\\test\\Accident20111129.csv"

precise,recall = judgeModel(probability,threshold, authenticPath, predictPath, saveDir=".\\result")
print("precise: ",precise,", recall: ",recall)
