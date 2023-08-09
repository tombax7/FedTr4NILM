import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# import argparse
from   WPuQ_Parser_FL    import *
from   config            import *
import json
from   Electricity_model import *
from   NILM_Dataloader   import *
from Trainer_FL import *
import pickle            as pkl
from client_utils import *


 


if __name__ == "__main__":
    
    args = get_args()   # Initialize arguments
    setup_seed(args.seed)

    results_root = Path(args.export_root).joinpath(args.dataset_code).joinpath(args.appliance_names[0])   
   
   
    #Initialize model
    model_avg = ELECTRICITY(args)
        

    
    create_dataset_and_evalmetrix(args) ## args.dis_cvs_files , args.metrics , args.client_length


    model_all = Partial_Client_Selection(args, model_avg)# Dict model_all to store every client's model

    args.years = [2018 , 2019]
    total_length = 0
    #Calculate total length of the series
    for single_client in args.dis_cvs_files:
        ds_parser = WPuQ_Parser(args , single_client)
        total_length += args.clients_length[single_client]
       


    # Initialize stats 
    x_mean = {}
    x_std = {}

    print('Training...')
    for communication_round in range(args.max_communication_rounds):    

        print("--------------------------Communication round:", communication_round,"-------------------------------")


        for single_client in args.dis_cvs_files:
            
            #Parser for each client, Trainer for every client
            
            ds_parser = WPuQ_Parser(args , single_client)
            args.clients_weightes[single_client] = args.clients_length[single_client] /total_length
            trainer = Trainer(args, ds_parser, model_all[single_client]) 

            
            
            if args.num_epochs > 0:
                
                trainer.train(args)  #run for every epoch of the specific client
               
   
   
       ##Testing Loop##
    args.validation_size = 1.

    args.years = [ 2020 ]

    print("Testing...")
    for single_client in args.dis_cvs_files:
        ds_parser      = WPuQ_Parser(args, single_client)
        
        trainer        = Trainer(args ,ds_parser, model_all[single_client])
        print("Testing client:", single_client)
        dataloader     = NILMDataloader(args, ds_parser)
        _, test_loader = dataloader.get_dataloaders()
        
        
        mre, mae, acc, prec, recall, f1, loss = trainer.test(test_loader)
        print( "Loss value of", single_client,":", loss)
        trainer.update_arguments_metrics(mae.tolist() ,mre.tolist() ,acc.tolist() ,prec.tolist() ,recall.tolist() ,f1.tolist() , args,'test')
    
   
    with open(results_root.joinpath("train_metrics_pers.json"), "w") as file:
        json.dump(args.train_metrics, file)
    
    with open(results_root.joinpath("val_metrics_pers.json"), "w") as file:
        json.dump(args.val_metrics, file)
    
    with open(results_root.joinpath("test_metrics_pers.json"), "w") as file:
        json.dump(args.test_metrics, file)

