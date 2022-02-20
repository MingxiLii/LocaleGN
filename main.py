import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import datetime
import h5py

import argparse
import wandb
import geopy.distance

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch_geometric.data import Data


from LocaleGN import Net
from Residual_GN import Res_GN
from config import DefaultConfig
opt = DefaultConfig()
from utils_tool import get_laplacian, Evaluation, generate_dataset_adj, loss_multi_step
from generate_PEMS_data import generate_PEMS_dataset_adj

from AttentionGN import attention_gn


#%%

parser = argparse.ArgumentParser(description='Experiments for traffic prediction')
parser.add_argument('--model_used', type=str, default='LocaleGN',
                        help='gn,res_gn, LocaleGN, GN_Ttransfomer, Ttransfomer_GN, Self_Attention_GN, Self_Attention_Res_GN')
parser.add_argument('--output_size', type=int, default=1,
                        help='the number snapshot graph for predcition, default as 1')
parser.add_argument('--forward_expansion', type= int, default=1)
parser.add_argument('--num_layer', type=int , default= 1)
parser.add_argument('--pre_epoch', type=int, default=2000,
                        help='epochs, default as 10000')
parser.add_argument('--epoch', type= int, default=3000)
parser.add_argument('--node_attr_size', type= int, default=12)
parser.add_argument('--pre_trained', type=bool, default = True)
parser.add_argument('--used_subset', type=bool, default = True)
parser.add_argument('--only_train_sub', type=bool, default=False)
parser.add_argument('--his_added', type=bool, default=True)
parser.add_argument("--n_his", type=int, default=12)
parser.add_argument('--edge_num_embedding', type = int, default=64)
parser.add_argument("--node_hidden_size", type=int, default=64)
parser.add_argument("--edge_hidden_size", type=int, default=64)
parser.add_argument("--global_hidden_size", type=int, default=64)
parser.add_argument("--in_channel", type=int, default= 1)
parser.add_argument("--heads", type=int , default=1 )
parser.add_argument("--reg_coeff", type=float, default=0)
parser.add_argument("--sim_coeff", type=float, default=0)
parser.add_argument('--sgd_lr',type=float, default=0.0000001, help = 'learning_rate')
parser.add_argument('--adam_lr',type=float, default=0.001, help = 'learning_rate')
parser.add_argument('--wdecay',type=float, default=0.0005, help = 'weight_decay')
parser.add_argument('--pre_momentum', type=float, default=0.99, help="momentum")
parser.add_argument('--fine_tuning_momentum', type=float, default=0.99, help="momentum")
parser.add_argument("--batch", type=int, default=12)
parser.add_argument('--tr_index', type=int, default=1512)
parser.add_argument('--val_index', type=int, default=1717)
parser.add_argument('--te_index', type=int, default=2016)
parser.add_argument('--PDE', type= str, default= "diff")
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--pre_data', type=str , default='PEMS08',
                    help='LA, HK, ST, PEMSD7, PEMS04, PEMS07, PEMS08')
parser.add_argument('--data',type=str , default='PEMS07',
                    help='LA, HK, ST, PEMSD7, PEMS04, PEMS07, PEMS08')
parser.add_argument('--data_percent', type=float, default = 0.2)
parser.add_argument('--use_existed_model', type=bool, default=True)
parser.add_argument('--exited_model_path', type=str, default= '.pt')
parser.add_argument('--train_dataset_path', type=str, default='~/deep_learning_implementation/Data/data_5_min/LA/los_speed.csv',
                        help='the path of training dataset file, los_speed.csv for LA, \
                            HKdata.csv for HK, and seattle_spd.csv for ST')
parser.add_argument('--train_adj_path', type=str,default='~/deep_learning_implementation/Data/data_5_min/LA/los_adj.csv',
                        help='the path of training adj file, los_adj.csv for LA, \
                                HKadj.csv for HK, and seattle_adj.csv for ST')
parser.add_argument('--pre_trained_path', type=str, default='gn_ex.pt')
parser.add_argument('--pre_optimizer_type', type = str, default='adam')
parser.add_argument('--fine_tuning_optimizer_type', type = str, default='sgd')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--skip_fine_tune', type=bool, default = True )
parser.add_argument('--trained_model_path', type=str, default='LocaleGN_hn12_out1_embed64_lr1e-06_wd0.0005_PDEdiff_eff0_PreTrue_PDataLA_SubTrue_DataST.pt')
args = parser.parse_args()
# project_name = args.model_used + "_pre_optimizer_" + str(args.pre_optimizer_type)  + '_fine_tuning_optimizer_'+ str(args.fine_tuning_optimizer_type)  + '_' + args.pre_data + "_" + args.data
project_name = "LocaleGN_trasfer_without_finetune_"+ args.pre_data + "_" + args.data
wandb.init(project= project_name, config=args)
api = wandb.Api()
#%% prepare parameters
if args.pre_trained== False:
    args.use_existed_model = False

if args.only_train_sub == True:
    args.pre_trained = False
    args.used_subset = True
    args.use_existed_model = False
if args.model_used == 'LocaleGN':
    args.his_added = False
device = args.device
node_attr_size = args.node_attr_size
output_size = args.output_size
forward_expansion = args.forward_expansion
edge_num_embedding = args.edge_num_embedding
edge_hidden_size = args.edge_hidden_size
node_hidden_size = args.node_hidden_size
global_hidden_size = args.global_hidden_size
in_channel = args.in_channel
heads = args.heads
sgd_lr = args.sgd_lr
adam_lr = args.adam_lr
weight_decay = args.wdecay
pre_momentum = args.pre_momentum
fine_tuning_momentum = args.fine_tuning_momentum
reg_coeff = args.reg_coeff
sim_coeff = args.sim_coeff
PDE = args.PDE
model_used = args.model_used

his_added = args.his_added
n_his = args.n_his
num_processing_steps = args.batch
tr_ind = args.tr_index
val_ind = args.val_index
te_ind = args.te_index
data_percent = args.data_percent

use_exsited_model = args.use_existed_model
pre_trained = args.pre_trained
pre_data = args.pre_data
pre_epoch = args.pre_epoch

data = args.data
Epoch = args.epoch
only_train_sub = args.only_train_sub
used_subset = args.used_subset
num_layer = args.num_layer

pre_optimizer_type = args.pre_optimizer_type
fine_tuning_optimizer_type = args.fine_tuning_optimizer_type

trained_model_path = args.trained_model_path

skip_fine_tune = args.skip_fine_tune

speed_list = ['HK', 'LA', 'ST' ]
pems_list = [ 'PEMSD7','PEMS04_speed', 'PEMS08_speed','PEMS04', 'PEMS07', 'PEMS08']
flow_list = ['PEMS04', 'PEMS07', 'PEMS08']

setting = '{}_hn{}_out{}_embed{}_lr{}_wd{}_PDE{}_eff{}_Pre{}_PData{}_Sub{}_Data{}'.format(args.model_used,  args.n_his,
            args.output_size, args.node_hidden_size, args.sgd_lr,
            args.wdecay, args.PDE, args.reg_coeff, args.pre_trained, args.pre_data, args.used_subset, args.data)

#%%
## create model
if model_used == 'GN':
    model = Res_GN(node_attr_size, edge_num_embedding, output_size, num_layer, device,
                   edge_hidden_size, node_hidden_size, global_hidden_size, add_residual= False)
if model_used=='Residual_GN':
    model = Res_GN(node_attr_size, edge_num_embedding, output_size, num_layer, device,
                   edge_hidden_size, node_hidden_size, global_hidden_size )

if model_used == 'LocaleGN': # node_attr_size=1 10 graphs to predict one
    model = Net(1, edge_num_embedding, output_size, num_layer, device, # inchanel = 1
                edge_hidden_size, node_hidden_size, global_hidden_size )

if model_used == "Self_Attention_GN":
    model = attention_gn(in_channel, node_attr_size, edge_num_embedding, output_size,  num_layer, device,
                         edge_hidden_size , node_hidden_size ,global_hidden_size, heads)


model.to(device)
wandb.watch(model, log_freq=100)
model.train()

## loss
pre_loss = nn.MSELoss()
## Optimizer
if pre_optimizer_type== 'adam' :
    pre_optimizer = optim.Adam(model.parameters(), lr = adam_lr, weight_decay = weight_decay)
else:
    pre_optimizer = optim.SGD(model.parameters(), lr=sgd_lr, momentum= pre_momentum)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

#%%
pre_losses_sup = []  # supervised loss
pre_losses_phy = []  # physics loss
pre_losses_tot = []  # total loss
pre_losses_val = []
pre_val_losses_sup = []
pre_used_timestamps = []
#### Pre_Training
if os.path.exists('/home/mingxi/GitHub/GN_pytorch/experiment/transfer_without_finetune/'+ trained_model_path +'.pt') == False:
    file_exsits = os.path.exists('/home/mingxi/GitHub/GN_pytorch/experiment/transfer_without_finetune/'+ trained_model_path +'.pt')
    print(os.path.exists('/home/mingxi/GitHub/GN_pytorch/experiment/transfer_without_finetune/'+ trained_model_path +'.pt'))

    if pre_trained == True and use_exsited_model == False :
        # prepare pre_train dataset
        dataset = [s for s in flow_list if s != data]
        if dataset in speed_list:
            X_1, sp_L_1, edgelist_1, edge_index_1, edge_attr_1, num_nodes_1, max_value_1 = generate_dataset_adj(dataset[0],device, data_used_subset=False)
            X_2, sp_L_2, edgelist_2, edge_index_2, edge_attr_2, num_nodes_2, max_value_2 = generate_dataset_adj(dataset[0],device, data_used_subset=False)
        else:
            X_1, sp_L_1, edgelist_1, edge_index_1, edge_attr_1, num_nodes_1, max_value_1 = generate_PEMS_dataset_adj(dataset[0], device)
            X_2, sp_L_2, edgelist_2, edge_index_2, edge_attr_2, num_nodes_2, max_value_2 = generate_PEMS_dataset_adj(dataset[1], device)

        for epoch in range(pre_epoch):
            #### Sample a starting timestep randomly
            if skip_fine_tune==True:
                tr_ind = 288
                val_ind = 576
            t = np.random.randint(num_processing_steps, tr_ind)  # low (inclusive) to high (exclusive)
            pre_used_timestamps.append(t)
            random_choice = random.choice([True, False])
            if random_choice == True:
                X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = X_1, sp_L_1, edgelist_1, edge_index_1, edge_attr_1, num_nodes_1, max_value_1
            else:
                X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = X_2, sp_L_2, edgelist_2, edge_index_2, edge_attr_2, num_nodes_2, max_value_2

            if his_added == True:
                input_graphs = [Data(x=torch.tensor(X[t + step_t - n_his + 1 : t + step_t + 1, :, 0:], dtype=torch.float32).transpose(0, 2).squeeze(0).to(device),
                                edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

            if his_added == False:
                input_graphs = [Data(x=torch.tensor(X[t + step_t, :, 0:], dtype=torch.float32, device=device),
                                edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

            input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)  # initial global_attr

            #### Passing the model
            output_tensors, time_derivatives, spatial_derivatives = model(input_graphs, sp_L, t, num_processing_steps, torch.tensor(0.001))

            if model_used != 'LocaleGN':
                # print(X[t  + 1: t + output_size + 1, :, :].shape)
                # print(output_tensors[0].shape)
                pre_loss_sup_seq = [torch.sum((output - torch.tensor(X[t + step_t + 1: t + step_t + output_size + 1, :, :].transpose(0, 2),
                                dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
                pre_loss_sup = sum(pre_loss_sup_seq) / len(pre_loss_sup_seq)
                ### Cosine_similarity
                pre_loss_sim_seq = [F.cosine_similarity(output.view(-1), torch.tensor(X[t + step_t + 1: t + step_t + output_size + 1, :, :].transpose(0, 2),
                                    dtype=torch.float32, device=device).view(-1), dim=0).mean() for step_t, output in enumerate(output_tensors)]
                pre_loss_sim = sum(pre_loss_sim_seq) / len(pre_loss_sim_seq)
            if model_used == 'LocaleGN':
                pre_loss_sup = torch.sum((output_tensors - torch.tensor(X[t + num_processing_steps, :, 0:], dtype=torch.float32, device=device)) ** 2)
                pre_loss_sim = F.cosine_similarity(output_tensors.view(-1),torch.tensor(X[t + num_processing_steps, :, 0:], dtype=torch.float32, device=device).view(-1), dim=0).mean()

            #### Physics rule
            pre_loss_phy_seq = [torch.sum((dt - ds) ** 2) for dt, ds in zip(time_derivatives, spatial_derivatives)]
            pre_loss_phy = sum(pre_loss_phy_seq) / len(pre_loss_phy_seq)

            pre_loss = pre_loss_sup + reg_coeff*pre_loss_phy + sim_coeff*pre_loss_sim
            print("Epoch:", epoch, "train loss:",pre_loss.item(), "loss_sup:", pre_loss_sup.item(), "loss_phy:", pre_loss_phy.item())

            pre_losses_sup.append(pre_loss_sup.item())
            pre_losses_phy.append(pre_loss_phy.item())
            pre_losses_tot.append(pre_loss.item())

            if epoch % 100 == 0 and epoch!= 0:
                wandb.log({"pre_train_loss": sum(pre_losses_tot)/len(pre_losses_tot)})
                pre_losses_tot.clear()


            #### Backward and optimize
            pre_optimizer.zero_grad()
            pre_loss.backward()
            pre_optimizer.step()


            if epoch % 100 == 0 and epoch != 0:
                for vt in range(tr_ind, val_ind - num_processing_steps):
                    if his_added == True:
                        input_graphs = [Data(x=torch.tensor(X[vt+step_t - n_his + 1 : vt + step_t + 1, :, 0:],dtype=torch.float32).transpose(0,2).squeeze(0).to(device),
                                        edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

                    if his_added == False:
                        input_graphs = [Data(x=torch.tensor(X[vt + step_t , :, 0:],dtype=torch.float32,device=device),
                                        edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

                    input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)  # initial global_attr

                    output_tensors, time_derivatives, spatial_derivatives = model(input_graphs, sp_L, vt, num_processing_steps, torch.tensor(0.001))

                    if model_used != 'LocaleGN':
                        pre_val_loss_sup_seq = [torch.sum((output- torch.tensor(X[vt + step_t + 1: vt + step_t + output_size + 1, :, :].transpose(0,2),
                            dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
                        pre_val_loss_sup = sum(pre_val_loss_sup_seq) / len(pre_val_loss_sup_seq)

                    if model_used == 'LocaleGN':
                        pre_val_loss_sup = torch.sum((output_tensors - torch.tensor(X[vt + num_processing_steps, :, 0:],dtype=torch.float32, device=device)) ** 2)

                    pre_losses_val.append(pre_val_loss_sup.item())

                    # print("Epoch:", epoch, "pre train valid loss:",pre_val_loss_sup.item())

                wandb.log({"pre_valid_loss": sum(pre_losses_val) / len(pre_losses_val)})
                pre_losses_val.clear()
        print("train loss", np.mean(pre_losses_tot) / num_nodes)
        print("val_loss", np.mean(pre_losses_val)/num_nodes)

        # torch.save(model.state_dict(), 'pre'+ setting+'.pt')
        torch.save(model.state_dict(), "transfer_without_finetune/"+ trained_model_path + '.pt')
#%%
### Fine tuning
if skip_fine_tune == False :
    tr_ind = args.tr_index
    val_ind = args.val_index
    print(tr_ind)
    print(val_ind)

    if used_subset  == True:
        tr_ind = round(tr_ind*data_percent)
    if pre_trained == True :
        if use_exsited_model == True :
            model.load_state_dict(torch.load())
            model.to(device)
            model.train()
        if use_exsited_model == False :
            model.load_state_dict(torch.load('pre'+setting+'.pt'))
            model.train()

    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = adam_lr, weight_decay = weight_decay)
    if fine_tuning_optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr= sgd_lr, momentum= fine_tuning_momentum)

    dataset = data
    if dataset in speed_list:
        X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = generate_dataset_adj(dataset, device, data_used_subset=False)
    if dataset in pems_list:
        X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = generate_PEMS_dataset_adj(dataset, device)

    losses_sup = []  # supervised loss
    losses_phy = []  # physics loss
    losses_tot = []  # total loss
    losses_val = []
    val_losses_sup = []
    used_timestamps = []
    for epoch_ in range(Epoch):
        #### Sample a starting timestep randomly
        t = np.random.randint(num_processing_steps, tr_ind)  # low (inclusive) to high (exclusive)
        used_timestamps.append(t)

        if his_added == True:
            input_graphs = [Data(x=torch.tensor(X[t + step_t - n_his + 1: t + step_t + 1, :, 0:], dtype=torch.float32).transpose(0, 2).squeeze(0).to(device),
                            edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

        if his_added == False:
            input_graphs = [Data(x=torch.tensor(X[t + step_t, :, 0:], dtype=torch.float32, device=device),
                            edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

        input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)  # initial global_attr

        #### Passing the model
        output_tensors, time_derivatives, spatial_derivatives = model(input_graphs, sp_L, t, num_processing_steps, torch.tensor(0.001))

        if model_used != 'LocaleGN':
            loss_sup_seq = [torch.sum((output - torch.tensor(X[t + step_t + 1: t + step_t + output_size + 1, :, :].transpose(0,2),
                            dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
            loss_sup = sum(loss_sup_seq) / len(loss_sup_seq)

            ### Cosine_similarity
            loss_sim_seq = [F.cosine_similarity(output.view(-1), torch.tensor( X[t + step_t + 1: t + step_t + output_size + 1, :, :].transpose(0, 2),
                            dtype=torch.float32, device=device).view(-1), dim=0).mean() for step_t, output in enumerate(output_tensors)]
            loss_sim = sum(loss_sim_seq) / len(loss_sim_seq)
        if model_used == 'LocaleGN':
            loss_sup = torch.sum((output_tensors - torch.tensor(X[t + num_processing_steps, :, 0:], dtype=torch.float32, device=device)) ** 2)
            loss_sim = F.cosine_similarity(output_tensors.view(-1),torch.tensor(X[t + num_processing_steps, :, 0:], dtype=torch.float32, device=device).view(-1), dim=0).mean()

        #### Physics rule
        loss_phy_seq = [torch.sum((dt - ds) ** 2) for dt, ds in zip(time_derivatives, spatial_derivatives)]
        loss_phy = sum(loss_phy_seq) / len(loss_phy_seq)

        loss = loss_sup + reg_coeff * loss_phy
        print("Epoch:", epoch_ , "train loss:",loss.item(), "loss_sup:", loss_sup.item(), "loss_phy:", loss_phy.item())

        losses_sup.append(loss_sup.item())
        losses_phy.append(loss_phy.item())
        losses_tot.append(loss.item())

        if epoch_ % 100 == 0 and epoch_ != 0 :

            wandb.log({"fine_tuning_train_loss": sum(losses_tot)/len(losses_tot)})
            losses_tot.clear()

        #### Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch_ % 100 == 0 and epoch_ != 0 :

            tr_ind = args.tr_index

            for vt in range(tr_ind, val_ind - num_processing_steps):
                if his_added == True:
                    input_graphs = [Data(x=torch.tensor(X[vt+step_t - n_his + 1 : vt + step_t + 1, :, 0:],dtype=torch.float32).transpose(0,2).squeeze(0).to(device),
                                    edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

                if his_added == False:
                    input_graphs = [Data(x=torch.tensor(X[vt + step_t , :, 0:],dtype=torch.float32,device=device),
                                    edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

                input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)  # initial global_attr

                output_tensors, time_derivatives, spatial_derivatives = model(input_graphs, sp_L, vt, num_processing_steps,torch.tensor(0.001))


                if model_used != 'LocaleGN':
                    val_loss_sup_seq = [torch.sum((output - torch.tensor(X[vt + step_t + 1: vt + step_t + output_size + 1, :, :].transpose(0,2),
                                        dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
                    val_loss_sup = sum(val_loss_sup_seq) / len(val_loss_sup_seq)

                if model_used == 'LocaleGN':
                    val_loss_sup = torch.sum((output_tensors - torch.tensor(X[vt + num_processing_steps, :, 0:],dtype=torch.float32, device=device)) ** 2)

                losses_val.append(val_loss_sup.item())

                # print("Epoch:", epoch_, "valid loss:",val_loss_sup.item())
            wandb.log({"fine_tuning_valid_loss": sum(losses_val) / (len(losses_val))})
            losses_val.clear()


    print("train loss", np.mean(losses_tot) / num_nodes)
    print("val_loss", np.mean(losses_val)/num_nodes)

    torch.save(model.state_dict(), setting+'.pt')

#%%
###Test dataset in LA, HK or ST
dataset = data
if dataset in speed_list:
    X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = generate_dataset_adj(dataset, device, data_used_subset=False)
if dataset in pems_list:
    X, sp_L, edgelist, edge_index, edge_attr, num_nodes, max_value = generate_PEMS_dataset_adj(dataset, device)
val_ind= args.val_index
te_ind = args.te_index
print(val_ind)
print(te_ind)

loss_one_step = []
loss_two_step = []
loss_three_step = []
losses_te = []
Target = np.zeros([num_nodes,  1])
Predict = np.zeros_like(Target)
Input_data = np.zeros_like(Target)

# model.load_state_dict(torch.load(setting+'.pt'))
# model.load_state_dict(model.state_dict(), "transfer_without_finetune/"+setting + '.pt')
model.load_state_dict(torch.load('transfer_without_finetune/'+ trained_model_path +'.pt'))
model.to(device)
model.eval()
with torch.no_grad():
    MAE, MAPE, RMSE = [], [], []
    for tt in range(val_ind, te_ind - num_processing_steps):
        print(tt)
        if his_added == True:
            input_graphs = [Data(x=torch.tensor(X[tt + step_t - n_his + 1: tt + step_t + 1, :, 0:], dtype=torch.float32).transpose(0,2).squeeze(0).to(device),
                            edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

        if his_added == False:
            input_graphs = [Data(x=torch.tensor(X[tt + step_t , :, 0:],dtype=torch.float32,device=device),
                            edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)) for step_t in range(num_processing_steps)]

        input_graphs[0].global_attr = torch.zeros((1, global_hidden_size), device=device)  # initial global_attr

        output_tensors, time_derivatives, spatial_derivatives = model(input_graphs, sp_L, tt, num_processing_steps, torch.tensor(0.001))

        # if model_used == "GN_Ttransfomer" or "Ttransfomer_GN":
        #     multi_loss = loss_multi_step(output_tensors,X, tt, output_size,num_nodes, his_added)
        #     loss_one_step.append(multi_loss[0])
        #     loss_two_step.append(multi_loss[1])
        #     loss_three_step.append(multi_loss[2])

        if model_used != 'LocaleGN':
            #### Test loss across processing steps.
            te_loss_sup_seq = [torch.sum((output - torch.tensor(X[tt + step_t + 1: tt + step_t + output_size + 1, :, :].transpose(0, 2),
                                dtype=torch.float32, device=device)) ** 2) for step_t, output in enumerate(output_tensors)]
            te_loss_sup = sum(te_loss_sup_seq) / len(te_loss_sup_seq)
            if model_used == "res_gn" or "gn":
                predict_value = torch.tensor(output_tensors[0]).squeeze(0).cpu()
            else:
                predict_value = torch.tensor(output_tensors[0]).cpu()
            target_value = torch.tensor(X[tt + 1, :, 0:]).cpu()
            input_value = torch.tensor(X[tt, :, 0:]).cpu()

        if model_used == 'LocaleGN':
            te_loss_sup = torch.sum((output_tensors - torch.tensor(X[tt + num_processing_steps, :, 0:],dtype=torch.float32, device=device)) ** 2)
            predict_value = torch.tensor(output_tensors).squeeze(0).cpu()
            target_value = torch.tensor(X[tt + num_processing_steps, :, 0:]).cpu()
            input_value = torch.tensor(X[tt+ num_processing_steps -1, :, 0:]).cpu()

        print("test loss:", te_loss_sup.item())
        wandb.log({"single_test_loss": te_loss_sup.item()})


    # print(np.mean(loss_one_step))
    # print(np.mean(loss_two_step))
    # print(np.mean(loss_three_step))

        losses_te.append(te_loss_sup.item())

        mae, mape, rmse = Evaluation.total((target_value*max_value).numpy().reshape(-1), (predict_value*max_value).numpy().reshape(-1))

        performance=[mae, mape, rmse]

        Predict = np.concatenate([Predict, torch.mul(predict_value, max_value) ], axis=1)
        Target = np.concatenate([Target, torch.mul(target_value, max_value) ], axis=1)
        Input_data = np.concatenate([Input_data, torch.mul(input_value, max_value) ], axis=1)

        MAE.append(performance[0])
        MAPE.append(performance[1])
        RMSE.append(performance[2])
    wandb.log({"test_loss": sum(losses_te) / len(losses_te)})
    wandb.log({"MAE": np.mean(MAE)})
    wandb.log({"MAPE": np.mean(MAPE)})
    wandb.log({"RMSE": np.mean(RMSE)})

    print("Performance:  MAE {:2.2f}    MAPE {:2.2f}%   RMSE {:2.2f}".format(np.mean(MAE), np.mean(MAPE), np.mean(RMSE)))

    Predict = np.delete(Predict, 0, axis=1)
    Target = np.delete(Target, 0, axis=1)
    Input_data = np.delete(Input_data, 0, axis=1)



#%%


if dataset == "LA":
    result_file = "LA_LocaleGN.h5"
elif dataset == "PEMSD7":
    result_file = "PEMSD7_LocaleGN.h5"
elif dataset == "ST":
    result_file = "ST_LocaleGN.h5"
elif dataset == "PEMS04":
    result_file = "PEMS04_LocaleGN.h5"
elif dataset == "PEMS07":
    result_file = "PEMS07_LocaleGN.h5"
else:
    result_file = "PEMS08_LocaleGN.h5"

file_obj = h5py.File(result_file, "w")

file_obj["predict"] = Predict  # [N, T, D]
file_obj["target"] = Target  # [N, T, D]
file_obj["input"] = Input_data



