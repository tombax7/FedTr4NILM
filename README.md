# Federated transformers for non-intrusive load monitoring of heatpumps
Pytorch implementation of federated transformers using WPuQ dataset

##Data

The .hdf5 files could be downloaded here: https://zenodo.org/record/5642902#.ZDFJ5tJBxkg

We use the 1min resolution datasets.
We change the name of the files in 2018.hdf5 , 2019.hdf5 , 2020.hdf5
 
 The folder structure in the data folder should be:
 

├── data
      └── WPuQ
             └──2018.hdf5
             │
             │
             └──2019.hdf5
                   .
                   .
      



##Training

We provide an end to end architecture to train and test transformer for NILM applications of heatpump load with federated learning.

The required packages to run the code could be found in requirements.txt.
Model , training and testing could be implemeneted by running the electricity_FL.py.

In the beginning config.py  provides all the hyperparameters that are used.Then the code creates the main model architecture that is used by all clients(ELECTRICITY(args).
The client_utils.py contains all the functions for the client creation and selection. Partial_client_selection(args,model) initialize the model architecture for all the clients.
 
 During the training process we create a parser (WPuQ_Parser_FL.py) for each client so as to parse the data of each client. Afterwards we create a Trainer(args, ds_parser, model_all[single_client]) (Trainer.py) for each parser(client) and the trainer.train(args) train each individual model with the data of each client separately. After every communication round the average_model(args, model_avg, model_all)
 implements the federated averaging and updates all the models.
 
 
##Testing


 After the training is complete we test the models of each clients in their own test dataset which is created by the WPuQ_Parser(args, single_client) and like the training mode we create a separate Trainer for each ds_parser. Finally we test each client separately and on his own test data through the trainer.test(test_loader). 

After the training and the testing the results are saved at the results/WPuQ/HEATPUMP we export the test_result_client_name.json where we store the plots for the ground truth and the prediction of each client and also the test_metrics.json , val_metrics.json, train_metrics.json where we store the metrics results for every client.


##Training configurations

The models are trained for 1 inner epoch each for 90 communication rounds with the hyperparameters that can be found in config.py
