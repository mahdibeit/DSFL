# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:59:46 2022

@author: Mahdi
"""
import numpy as np
from RandomSelection import Random_Selection
from LayerWiseSparsification import layerSparsification

def Flatten(w_w):
    flatten_weighted_weights=np.array([])
    map_shape={"Seperation":[0], "Shape":[]}  
    tot=0
    for  layers in w_w:
        temp=1
        flatten_weighted_weights=np.append(flatten_weighted_weights, layers.flatten())
        # print(layers.shape)
        map_shape["Shape"].append(layers.shape)
        for i in layers.shape:
            temp=temp*i
        tot+=temp
        map_shape["Seperation"].append(tot)

    return flatten_weighted_weights, map_shape


def DeFlatten(f_w_w, map_shape):
    out=[]
    for idx, shape in enumerate(map_shape["Shape"]):
        out.append(f_w_w[map_shape["Seperation"][idx]:map_shape["Seperation"][idx+1]].reshape(shape))
    for  layers in out:
        # print("outputcheck",layers.shape)
        pass
    return out




def alastor(Weights, history):
    
    flatten_weighted_weights= [Flatten(user)[0] for user in Weights[:]]
    map_shape=Flatten(Weights[0])[1]

    new_flatten_weighted_weights= Random_Selection(flatten_weighted_weights[:], history)

    output=[DeFlatten(user, map_shape) for user in new_flatten_weighted_weights]
    return output
    

    
    
    