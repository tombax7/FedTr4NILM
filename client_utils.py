import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
import torch
from torchvision import transforms
from config import *

from Electricity_model import *




def create_dataset_and_evalmetrix(args):

    
    if args.dataset_code == 'WPuQ':
        file = np.array(args.house_indicies).astype(str)
        for i in range(len(file)):
            file[i] = 'SFH' + file[i]
        
        args.dis_cvs_files = list(file)
    
    args.clients_length = {}
    

    ## Initialize metrics for Train, Validation, Accuracy
    metrics = ['mae' ,'mre', 'acc', 'precision', 'recall', 'f1', 'loss' ]
    for client in args.dis_cvs_files:
            args.train_metrics[client]  = {}
            args.test_metrics[client]   = {}
            args.val_metrics[client]    = {}
    
    for client in args.dis_cvs_files:
            for metric in metrics:
                args.train_metrics[client][metric]  = []
                args.test_metrics[client] [metric]  = []
                args.val_metrics[client]  [metric]  = []



    



def Partial_Client_Selection(args, model):

 
    args.proxy_clients = args.dis_cvs_files
    args.num_local_clients =  len(args.dis_cvs_files)# update the true number of clients
    
    # Generate model for each client
    model_all = {}
   
    args.learning_rate_record = {}

    for proxy_single_client in args.proxy_clients:
        if torch.cuda.is_available():
            model_all[proxy_single_client] = deepcopy(model).cuda()
        else:
            model_all[proxy_single_client] = deepcopy(model).cpu()
             
    args.clients_weightes = {}
   
    return model_all





def average_model(args,model_avg,model_all):
   
    
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())  

   
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
      
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
            if client == 0:
                tmp_param_data = dict(model_all[single_client].named_parameters())[      
                                     name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(model_all[single_client].named_parameters())[
                                     name].data * single_client_weight               
        params[name].data.copy_(tmp_param_data) 
    print('Update each client model parameters----')

    for single_client in args.proxy_clients:
        tmp_params = dict(model_all[single_client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)
    


