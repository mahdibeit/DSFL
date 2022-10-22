# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:03:34 2022

@author: Mahdi
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from FindingKSpeech import findK


def layerSparsification(flatten_weights, history, oldseperation):
    """ Input: flattern_weighted_weights--> List __> with size of number of users
                In each index of this list is a flatten numpy array with size for instane M
                                            
    
        Output: --> Should be a list with size of number of users
                    and each index of this list should be a numpy array with size M            
    """
    # Set parameters
    iteration = history.round - 1 # global round
    num_user = len(flatten_weights)
    dynamic_sparsification = True  # If false we use the minimum of K
    # k_list = [len(flatten_weights[0])] * num_user
    with open('K_alpha=10_gamma=10_test=0.pickle', 'rb') as f:
        k_list = pickle.load(f)
    seperation = oldseperation[::2]
    print(seperation)
    # k_list =[[300, 6, 200, 16, 96, 20, 40, 10, 30, 10]] * num_user
    # k_list =[[150, 6, 60, 16, 30, 20, 20, 10, 30, 10]] * num_user
    # k_list =[[200, 6, 130, 16, 196, 20, 110, 10, 30, 10]] * num_user
    # k_list =[[400, 6, 300, 16, 6, 0, 0, 0, 0, 0]] * num_user
    # k_list =[[30, 6, 40, 16, 76, 20, 200, 30, 300, 10]] * num_user
    # k_list =[[79, 4, 192, 3, 195, 3, 152, 7, 90, 3]] * num_user
    # if not dynamic_sparsification:
    #     print("Using Fixed Sparsification")
    #     k_list = [min(k_list)] * num_user
    # else:
    #     print("Using Dynamic Sparsification")
    # print(k_list)
    
    # Setting K
    # k_list = [400] * 10
    if iteration > 1:
        cka = findK()
    # k_list = findK(728)
    print(k_list)
    # print(cka)
    
    # Testing
    # for idx, user in enumerate(range(num_user)):
    #     print("actual", idx, flatten_weights[user][:10])
        


    # Calculate model difference
    model_difference = [user_weights - history.globals[iteration]
                        for user_weights in flatten_weights]

    # Calculate the error accumulated models
    model_difference_AccError = [model_difference[i] + history.error[iteration][i] for i in range(num_user)]

    # Calculate the mask list
    # mask_list = [np.array([0]*len(flatten_weights[0])) for _ in range(num_user)]
    # for layer in range(len(seperation)-1):
    #     for user in range(num_user):
    #         if k_list[user] != 0:
    #             layer_model = np.absolute(model_difference_AccError[user][seperation[layer]:seperation[layer+1]])
    #             for index in np.argpartition(layer_model, -k_list[user][layer])[-k_list[user][layer]:].tolist():
    #                 mask_list[user][seperation[layer]+index] = 1
    
    # """New method"""
    # mult_list = [np.array([0.0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     for layer in range(len(seperation)-1):
    #         for index in range (seperation[layer+1]-seperation[layer]):
    #             mult_list[user][seperation[layer] + index] = cka[user][layer]
                
    """New New method"""
    mult_list = [np.array([1.0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    if iteration > 2:
        for user in range(num_user):
            for layer in range(len(seperation)-1):
                for index in range (seperation[layer+1]-seperation[layer]):
                    mult_list[user][seperation[layer] + index] = np.random.binomial(1, 1-cka[user][layer])


    # Calculate the mask list
    mask_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    for user in range(num_user):
        if k_list[user] != 0:
            mult = np.multiply(model_difference_AccError[user], mult_list[user])
            for index in np.argpartition(np.absolute(mult), -k_list[user])[-k_list[user]:].tolist():
                mask_list[user][index] = 1
                
    # """New Very new method"""
    # mult_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     for layer in range(len(seperation)-1):
    #         for index in range (seperation[layer+1]-seperation[layer]):
    #             mult_list[user][seperation[layer] + index] = np.random.binomial(1, 1)


    # # Calculate the mask list
    # mask_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     if k_list[user] != 0:
    #         for index in np.argpartition(np.absolute(model_difference_AccError[user]), -k_list[user])[-k_list[user]:].tolist():
    #             mask_list[user][index] = 1
    #     mask_list[user] = np.multiply(mask_list[user], mult_list[user])
    
    
    print([sum(mask_list[i]) for i in range(num_user)])
    # Calculate the model for sending
    model_to_send = [np.multiply(mask_list[user], model_difference_AccError[user]) for user in range(num_user)]
    
    # Plot
    # for layer in range(len(seperation)-1):
    #     plt.figure()
    #     if layer == 4:
    #         plt.figure(figsize = (50, 5))
    #     plt.bar([i for i in range(seperation[layer+1] - seperation[layer])] , model_to_send[0][seperation[layer]:seperation[layer+1]])
    #     plt.savefig(f"layerwisefigs\{iteration}{layer}.png")
    
    # Update the error
    new_error = [np.multiply(np.array([1]*len(mask_list[0]))- mask_list[user], model_difference_AccError[user]) for user in range(num_user)]
    history.updateError(new_error)

    # Average the model differences
    global_model_difference = model_to_send[0]
    for user in range(1, num_user):
        global_model_difference += model_to_send[user]
    global_model_difference /= len(flatten_weights)

    # Calculate the next global round
    global_model =  history.globals[iteration] + global_model_difference
    history.updateGlobal(global_model)

    # Dummy global model
    global_model = [global_model for _ in range(num_user)]

    return global_model 
