import osmnx as ox
import os

G = ox.graph_from_place('Beijing, China',network_type = 'drive',which_result=2,retain_all=True)
ox.save_graphml(G, filename='BeijingStreetMap') #将数据储存到文件BeijingStreetMap中