import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import h5py
import numpy as np
from config import *
from pathlib import Path
import pandas as pd
from NILM_Dataset import *
from   NILM_Dataloader   import *
from Pretrain_Dataset import *
from matplotlib import pyplot as plt
from Electricity_model import *
from client_utils import *
from Trainer_FL import *


class WPuQ_Parser:

    def __init__(self,args, single_client, stats = None):
        
        self.window_size     = args.window_size
        self.house_indicies  = args.house_indicies
        self.years           = args.years                     # train [2018, 2019]  testing [2020]
        self.window_stride   = args.window_stride
        self.val_size        = args.validation_size           #train 0.1       testing 1.
        self.normalize       = args.normalize
        self.appliances_names = args.appliance_names
        self.cutoff        =  [args.cutoff[appl]    for appl in ['HOUSEHOLD']+args.appliance_names]
        self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]
        self.single_client = single_client
    
       
        
       
        
        self.data_location = args.WPuQ_location
        
        self.x , self.y      = self.load_data(args)
        
        
        if self.normalize == 'mean':
            if stats is None:
                self.x_mean = np.mean(self.x)
                self.x_std  = np.std(self.x)
            else:
                self.x_mean,self.x_std = stats
            self.x = (self.x - self.x_mean) / self.x_std
        elif self.normalize == 'minmax':
            if stats is None:
                self.x_min = min(self.x)
                self.x_max = max(self.x)
            else:
                self.x_min,self.x_max = stats
            self.x = (self.x - self.x_min)/(self.x_max-self.x_min)
            
        self.status = self.compute_status(self.y)
        

    def load_data(self,args):
        directory = Path(self.data_location)

      # assert self.single_client in args.dis_cvs_files # [ 'SFH3' , 'SFH4', 'SFH5', 'SFH7', 'SFH9' , 'SFH10', 'SFH12', 'SFH14', 'SFH16', 'SFH18', 'SFH19', 'SFH21', 'SFH22', 'SFH23', 'SFH27', 'SFH28', 'SFH29', 'SFH30', 'SFH32', 'SFH34', 'SFH36', 'SFH38', 'SFH39']
        
        
            
        for year in self.years:
            f = h5py.File(directory.joinpath(str(year) + '.hdf5'),'r+')
            
            
            data_target = np.array(f['NO_PV'][self.single_client][self.appliances_names[0]]['table'])
            data_aggr   = np.array(f['NO_PV'][self.single_client]['HOUSEHOLD']['table'])

            

            
            df_target = pd.DataFrame(data_target)

            df_aggr =   pd.DataFrame(data_aggr)

            df_aggr['index']   = pd.to_datetime(df_aggr['index'],unit = 's') 

            df_target['index'] = pd.to_datetime(df_aggr['index'],unit = 's') 

            df_target.rename(columns={'P_TOT':'P_TOT_HEATPUMP'}, inplace = True)

            data_temp  = pd.merge(df_aggr[['index' , 'P_TOT']], df_target[['index', 'P_TOT_HEATPUMP']], how='inner', on='index') 
            
            data_temp.rename(columns = {'index' : 'Date'}, inplace = True)

            data_temp.set_index('Date')
            data_temp = data_temp.dropna().copy()        
            
            
            
            
            
            data_temp['P_TOT'] = data_temp['P_TOT'] + data_temp['P_TOT_HEATPUMP']
            val_end =  int(self.val_size * len(data_temp)) 
    

            train_data_temp   = data_temp.iloc [ val_end : , :].copy()
            val_data_temp     = data_temp.iloc [ : val_end , :].copy()
            
            
            if (year != 2019):
            
                train_data = train_data_temp
                val_data   = val_data_temp    
                
    
            else:
                train_data =  pd.concat( [train_data , train_data_temp], ignore_index= True) 
                val_data   =  pd.concat([val_data , val_data_temp], ignore_index= True)  
               

            


            
            
            
        
        
            
        entire_data_temp = pd.concat([val_data , train_data], ignore_index= True) #entire data = train + validation data for specific client or test data for specific client
            

        
      

       
        entire_data_temp.set_index('Date')
       
        cols = ['P_TOT', 'P_TOT_HEATPUMP']
        entire_data_temp = entire_data_temp[cols]
        entire_data_temp[entire_data_temp['P_TOT'] < 5] = 0 
          
        
        
        
        
        entire_data                                     = entire_data_temp[entire_data_temp['P_TOT'] > 0] 
     
        
        entire_data                                     = entire_data.clip([0] * len(entire_data.columns), self.cutoff , axis=1) 

        args.clients_length[self.single_client] =int((len(entire_data['P_TOT']) * (1 - self.val_size)))
        
        #print(entire_data)
        return entire_data.values[:, 0].astype(np.float64), entire_data.values[:, 1].astype(np.float64)
        


    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x)) 
        
        val = NILMDataset(self.x[:val_end],
                          self.y[:val_end],
                          self.status[:val_end],
                          self.window_size,
                          self.window_size    #non-overlapping windows
                          )

        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_stride,
                            self.window_size
                            
                            )
        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))

        val     = NILMDataset(self.x[:val_end],
                               self.y[:val_end],
                               self.status[:val_end],
                               self.window_size,
                               self.window_size
                             )
        train   = Pretrain_Dataset(self.x[val_end:],
                                   self.y[val_end:],
                                   self.status[val_end:],
                                   self.window_size,
                                   self.window_stride,
                                   mask_prob=mask_prob
                                   )
        return train, val        
    
    def compute_status(self, data):  #Ground truth status
    
    

        initial_status = data >= self.threshold  #Real data 0-off  1-on
        status_diff    = np.diff(initial_status) #When i have a change of status
        events_idx     = status_diff.nonzero()   #events index = timestep where i got a change in status
        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1    #next index


        if initial_status[0]:  #if initial status is on place 0 before to get the event
            events_idx = np.insert(events_idx, 0, 0)# Adds off-status in the beginning

        if initial_status[-1]:  #if last status is on add last index (525600) in the events
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)
        
        events_idx     = events_idx.reshape((-1, 2)) #1st row when i turn on the appliance, 2nd row when i turned off the appliance
        on_events      = events_idx[:, 0].copy() #on events 1st row 
        off_events     = events_idx[:, 1].copy() #off events 2nd row
      
        
        assert len(on_events) == len(off_events) 

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1] #counts the off duration between an on change and the previous off state
            off_duration = np.insert(off_duration, 0, 1000)#adds a 1000sec off duration in the beginning
           
            on_events    = on_events[off_duration > self.min_off[0]] #When off_duration is larger than the min off duration of the appliance  is an on event
            off_events   = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= self.min_on[0]]
            off_events   = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = data.copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1         #from on event to off event place on status
        status = temp_status
        
        return status       










