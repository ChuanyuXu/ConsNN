import time

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from torch import nn, tensor
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
# download data
Cap = pd.read_csv('D:\\data\\Infomation.csv',encoding="gbk").iloc[:, 2]
dataFile = 'D:\\data\\trainconsdata'
# the trainconsdata should include the inputs, targets, ond constrains samples series of test and tain set.
data = scio.loadmat(dataFile)
Cap = np.array(Cap).reshape(Cap.shape)
test_inputs0=data['test_input']
test_targets0=data['test_target']
test_constrain0=data['test_constrain']
train_inputs0=data['trian_input']
train_targets0=data['train_target']
train_constrain0=data['train_constrain']
# define device
device=torch.device("cuda")
device0=torch.device("cpu")
# the network include Constraint learning module, Constraint mapping module, Constraint reconciliation module, Residual module
# data processing（normalize）
train_targets=np.zeros([np.size(train_targets0,axis=0),np.size(train_targets0,axis=1)])
test_targets=np.zeros([np.size(test_targets0,axis=0),np.size(test_targets0,axis=1)])
train_inputs=np.zeros([np.size(train_inputs0,axis=0),np.size(train_inputs0,axis=1)])
test_inputs=np.zeros([np.size(test_inputs0,axis=0),np.size(test_inputs0,axis=1)])
for i in range(np.size(train_inputs0,axis=1)):
    train_inputs[:,i]=train_inputs0[:,i]/Cap[i]
    train_targets[:,i]=train_targets0[:,i]/Cap[i]
    test_inputs[:,i]=test_inputs0[:,i]/Cap[i]
    test_targets[:,i]=test_targets0[:,i]/Cap[i]
train_constrain = train_constrain0/sum(Cap)
test_constrain = test_constrain0/sum(Cap)
#############################—————————————————————————————————————————————##############################################
#############################———————————Constraint learning module————————##############################################
#############################—————————————————————————————————————————————##############################################
## define model
class mlp_conslearning(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(mlp_conslearning, self).__init__()
        self.model1 = Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 30),
            nn.Sigmoid(),
            nn.Linear(30, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, output_size),
        )

    def forward(self, x):
        y_res = self.model1(x)
        y=x+y_res
        return y
#
input_size = np.size(train_inputs,axis=1)
hidden_size1 = 40
output_size = np.size(train_targets,axis=1)
model_conslearning = mlp_conslearning(input_size, hidden_size1, output_size)
model_conslearning = model_conslearning.to(device)  # gpu加速

class RelaxationLoss(nn.Module):
    def __init__(self):
        super(RelaxationLoss, self).__init__()
        return

    def forward(self, outputs, targets, train_Constrain,Cap_sum,Cap00,k):
        loss1 = (targets - outputs) ** 2
        loss1 = torch.mean(loss1, dim=0)
        Cons_loss = (train_Constrain*Cap_sum-((torch.sum(outputs*Cap00,dim=1)).reshape(train_Constrain.size(dim=0),1)))/Cap_sum
        loss2 = (Cons_loss ** 2)*k
        loss2 = torch.mean(loss2)
        loss = torch.mean(loss1 + loss2)
        return loss

loss_fn_Relaxation = RelaxationLoss()
loss_fn_Relaxation = loss_fn_Relaxation.to(device)
# optimizer
learning_rate= 1e-1
optimizer = torch.optim.SGD(model_conslearning.parameters(), lr=learning_rate)


# prepare data
Train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
Train_targets = torch.tensor(train_targets, dtype=torch.float32)
Test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
Train_Constrain = torch.tensor(train_constrain, dtype=torch.float32)
Train_data = Data.TensorDataset(Train_inputs, Train_targets,Train_Constrain)
Cap_sum=torch.tensor(np.sum(Cap), dtype=torch.float32)
Cap00=torch.tensor(Cap, dtype=torch.float32)

# train
# Record the step of training
total_train_step = 0
# Record the step of testing
total_test_step = 0
# Number of epoch
epoch_learning = 10000

## Relaxation
# epoch
epoch__Relaxation = 30
# Multiplier
lamb=1
# subgradient
SubGrad=0
# StepSize
StepSize=0
SubGrad_list=list()
loss_UP_list=list()
loss_lamb_list=list()
StepSize_list=list()
lamb_list=list()
# add tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()
for j in range(epoch__Relaxation):

    for i in range(epoch_learning):
        # start training
        model_conslearning.train()
        Train_Constrain= Train_Constrain.to(device)
        Train_inputs = Train_inputs.to(device)
        Train_targets = Train_targets.to(device)
        Cap00=Cap00.to(device)
        Cap_sum = Cap_sum.to(device)
        Train_outputs = model_conslearning(Train_inputs)
        # loss
        loss = loss_fn_Relaxation(Train_outputs, Train_targets, Train_Constrain, Cap_sum, Cap00, lamb)
        if loss< loss_fn_Relaxation(Train_inputs/(torch.sum(Train_inputs*Cap00,dim=1).reshape(Train_outputs.size(dim=0),1))*Train_Constrain*Cap_sum, Train_targets, Train_Constrain, Cap_sum, Cap00, lamb):#tensor(0.04, device='cuda:0'):
            print('___________________________epoch__Relaxation: {}'.format(j))
            break
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss_conslearning", loss.item(), total_train_step)
    # update Multiplier
    with torch.no_grad():
        SubGrad= torch.mean(((Train_Constrain*Cap_sum-((torch.sum(Train_outputs*Cap00,dim=1)).reshape(Train_Constrain.size(dim=0),1)))/Cap_sum)** 2)
        # Using the base forecasting processed by the constraint learning module as feasible solutions for the dual problem
        loss_UP = loss_fn_Relaxation(Train_inputs/(torch.sum(Train_inputs*Cap00,dim=1).reshape(Train_outputs.size(dim=0),1))*Train_Constrain*Cap_sum, Train_targets, Train_Constrain, Cap_sum, Cap00, lamb)
        loss_lamb = loss_fn_Relaxation(Train_outputs, Train_targets, Train_Constrain, Cap_sum, Cap00, lamb)
        StepSize=2*(loss_UP-loss_lamb)/(SubGrad**2)
        lamb=lamb+SubGrad*StepSize
        #
        SubGrad_list.append(SubGrad)
        loss_UP_list.append(loss_UP)
        loss_lamb_list.append(loss_lamb)
        StepSize_list.append(StepSize)
        lamb_list.append(lamb)
        print('___________________________epoch__Relaxation: {}'.format(j))

writer.close()
model_conslearning.eval()
# forecasting
Test_inputs = Test_inputs.to(device)

with torch.no_grad():
    Test_outputs_conslearning = model_conslearning(Test_inputs)
    Test_outputs_conslearning = Test_outputs_conslearning.to(device0)
    test_outputs0_conslearning = np.array(Test_outputs_conslearning).reshape(np.size(test_targets, axis=0),
                                                  np.size(test_targets, axis=1))
test_outputs_conslearning=np.zeros((np.size(test_targets, axis=0),
                      np.size(test_targets, axis=1) ))
for i in range(np.size(train_inputs,axis=1)):
    test_targets[:, i] = test_targets[:, i] * Cap[i]
    test_outputs_conslearning[:, i] = test_outputs0_conslearning[:, i] * Cap[i]


# calculating rmse
test_inputs=test_inputs*(Cap.reshape([1,np.size(Cap)]))
rmse_test_ConslearningResult=(np.mean((test_targets-test_outputs_conslearning)**2,axis=0))**0.5/Cap
rmse_test_OrigResult=(np.mean((test_targets-test_inputs0)**2,axis=0))**0.5/Cap
rmse_test_diffO2C_conslearning=rmse_test_OrigResult-rmse_test_ConslearningResult

#
train_inputs=train_inputs*(Cap.reshape([1,np.size(Cap)]))
train_targets=train_targets*(Cap.reshape([1,np.size(Cap)]))
train_constrain=train_constrain*sum(Cap)
with torch.no_grad():
    Train_outputs0_conslearning = Train_outputs.to(device0)
    train_outputs_conslearning=np.array(Train_outputs0_conslearning)
train_outputs_conslearning=train_outputs_conslearning*(Cap.reshape([1,np.size(Cap)]))

rmse_train_ConslearningResult=(np.mean((train_targets-train_outputs_conslearning)**2,axis=0))**0.5/Cap
rmse_train_OrigResult=(np.mean((train_targets-train_inputs0)**2,axis=0))**0.5/Cap
rmse_train_diffO2C_conslearning=rmse_train_OrigResult-rmse_train_ConslearningResult


a_conslearning=np.mean(rmse_test_diffO2C_conslearning)
b_conslearning=np.mean(rmse_train_diffO2C_conslearning)

#############################—————————————————————————————————————————————##############################################
#############################———————————Constraint mapping module—————————##############################################
#############################—————————————————————————————————————————————##############################################
print('#############################———————————         Constraint mapping module      —————————##############################################')
train_outputs_consmapping=train_outputs_conslearning/(np.sum(train_outputs_conslearning,axis=1).reshape(np.size(train_outputs_conslearning,axis=0),1))*train_constrain
test_outputs_consmapping=test_outputs_conslearning/(np.sum(test_outputs_conslearning,axis=1).reshape(np.size(test_outputs_conslearning,axis=0),1))*test_constrain*np.sum(Cap)
rmse_train_consmappingResult=(np.mean((train_targets-train_outputs_consmapping)**2,axis=0))**0.5/Cap
rmse_test_consmappingsResult=(np.mean((test_targets-test_outputs_consmapping)**2,axis=0))**0.5/Cap
rmse_train_diffO2C=rmse_train_OrigResult-rmse_train_consmappingResult
rmse_test_diffO2C=rmse_test_OrigResult-rmse_test_consmappingsResult
a_consmapping=np.mean(rmse_test_diffO2C)
b_consmapping=np.mean(rmse_train_diffO2C)

#############################—————————————————————————————————————————————##############################################
#############################———————————Constraint reconciliation module—————————##############################################
#############################—————————————————————————————————————————————##############################################
print('#############################———————————         Constraint reconciliation module      —————————##############################################')

# data processing
train_inputs_reconcile0=train_outputs_consmapping
test_inputs_reconcile0=test_outputs_consmapping

# normalized
train_targets_reconcile=np.zeros([np.size(train_targets0,axis=0),np.size(train_targets0,axis=1)])
test_targets_reconcile=np.zeros([np.size(test_targets0,axis=0),np.size(test_targets0,axis=1)])
train_inputs_reconcile=np.zeros([np.size(train_inputs0,axis=0),np.size(train_inputs0,axis=1)])
test_inputs_reconcile=np.zeros([np.size(test_inputs0,axis=0),np.size(test_inputs0,axis=1)])
for i in range(np.size(train_inputs_reconcile0,axis=1)):
    train_inputs_reconcile[:,i]=train_inputs_reconcile0[:,i]/Cap[i]
    train_targets_reconcile[:,i]=train_targets0[:,i]/Cap[i]
    test_inputs_reconcile[:,i]=test_inputs_reconcile0[:,i]/Cap[i]
    test_targets_reconcile[:,i]=test_targets0[:,i]/Cap[i]
train_constrain_reconcile = train_constrain0/sum(Cap)
test_constrain_reconcile = test_constrain0/sum(Cap)

## define model
class mlp_reconcile(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(mlp_reconcile, self).__init__()
        self.model1 = Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 30),
            nn.Sigmoid(),
            nn.Linear(30, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, output_size),
        )

    def forward(self, x):
        y_res = self.model1(x)
        y=x[:,0:-1]+y_res
        return y
#
input_size = np.size(train_inputs_reconcile,axis=1)
hidden_size1 = 40
output_size = np.size(train_targets_reconcile,axis=1)-1
model_reconcile = mlp_reconcile(input_size, hidden_size1, output_size)
model_reconcile = model_reconcile.to(device)

class ConstraintsLoss(nn.Module):
    def __init__(self):
        super(ConstraintsLoss, self).__init__()
        return

    def forward(self, outputs, targets, train_Constrain,ConsTargets,Cap_sum,Cap00,Cap_end,k):
        loss1 = (targets - outputs) ** 2
        loss1 = torch.mean(loss1, dim=0)
        ConsOutputs=(train_Constrain*Cap_sum-((torch.sum(outputs*Cap00,dim=1)).reshape(train_Constrain.size(dim=0),1)))/Cap_end
        loss2 = ((ConsTargets - ConsOutputs) ** 2)/k
        loss2 = torch.mean(loss2)
        loss = torch.mean(loss1 + loss2)
        return loss

loss_fn = ConstraintsLoss()
loss_fn = loss_fn.to(device)
# optimizer
learning_rate1= 1e-1
learning_rate2= 1e-2
learning_rate3= 1e-3
learning_rate4= 1e-3
optimizer = torch.optim.SGD(model_reconcile.parameters(), lr=learning_rate1)

# data prepare
Train_inputs = torch.tensor(train_inputs_reconcile, dtype=torch.float32)
Train_targets = torch.tensor(train_targets_reconcile, dtype=torch.float32)
Test_inputs = torch.tensor(test_inputs_reconcile, dtype=torch.float32)
Train_Constrain = torch.tensor(train_constrain_reconcile, dtype=torch.float32)
Cap_sum=torch.tensor(np.sum(Cap), dtype=torch.float32)
Cap00=torch.tensor(Cap[0:-1], dtype=torch.float32)
Cap_end=torch.tensor(Cap[-1], dtype=torch.float32)


# train
# Record the step of training
total_train_step = 0
# Record the step of testing
total_test_step = 0
# training epoch
epoch1 = 10000
epoch2 = 10000
epoch3 = 10000
epoch4 = 20000
k1=100
k2=10
k3=3
k4=1
# add tensorboard
start_time = time.time()

for i in range(epoch1):
    # start training
    model_reconcile.train()
    Train_Constrain= Train_Constrain.to(device)
    Train_inputs = Train_inputs.to(device)
    Train_targets = Train_targets.to(device)
    Cap00=Cap00.to(device)
    Cap_sum = Cap_sum.to(device)
    Cap_end = Cap_end.to(device)
    Train_outputs = model_reconcile(Train_inputs)
    # loss function
    loss = loss_fn(Train_outputs, Train_targets[:, 0:-1], Train_Constrain, Train_targets[:, -1].reshape([Train_targets.size(dim=0),1]), Cap_sum, Cap00, Cap_end,k1)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_train_step = total_train_step + 1
    if total_train_step % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_reconcile", loss.item(), total_train_step)
optimizer = torch.optim.SGD(model_reconcile.parameters(), lr=learning_rate2)
for i in range(epoch2):
    # start training
    model_reconcile.train()
    Train_Constrain = Train_Constrain.to(device)
    Train_inputs = Train_inputs.to(device)
    Train_targets = Train_targets.to(device)
    Train_outputs = model_reconcile(Train_inputs)
    # loss function
    loss = loss_fn(Train_outputs, Train_targets[:, 0:-1], Train_Constrain,
                   Train_targets[:, -1].reshape([Train_targets.size(dim=0), 1]), Cap_sum, Cap00, Cap_end,k2)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_train_step = total_train_step + 1
    if total_train_step % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_reconcile", loss.item(), total_train_step)
optimizer = torch.optim.SGD(model_reconcile.parameters(), lr=learning_rate3)
for i in range(epoch3):
    # start training
    model_reconcile.train()
    # for Train_inputs_dataloader,Train_targets_dataloader,Train_Constrain_dataloader in Train_data_dataloader0:
    Train_Constrain = Train_Constrain.to(device)
    Train_inputs = Train_inputs.to(device)
    Train_targets = Train_targets.to(device)
    # for data in train_inputs:
    Train_outputs = model_reconcile(Train_inputs)
    # loss function
    loss = loss_fn(Train_outputs, Train_targets[:, 0:-1], Train_Constrain,
                   Train_targets[:, -1].reshape([Train_targets.size(dim=0), 1]), Cap_sum, Cap00, Cap_end,k3)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_train_step = total_train_step + 1
    if total_train_step % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_reconcile", loss.item(), total_train_step)
optimizer = torch.optim.SGD(model_reconcile.parameters(), lr=learning_rate4)
for i in range(epoch4):
    # start training
    model_reconcile.train()
    # for Train_inputs_dataloader,Train_targets_dataloader,Train_Constrain_dataloader in Train_data_dataloader0:
    Train_Constrain = Train_Constrain.to(device)
    Train_inputs = Train_inputs.to(device)
    Train_targets = Train_targets.to(device)
    # for data in train_inputs:
    Train_outputs = model_reconcile(Train_inputs)
    # loss function
    loss = loss_fn(Train_outputs, Train_targets[:, 0:-1], Train_Constrain,
                   Train_targets[:, -1].reshape([Train_targets.size(dim=0), 1]), Cap_sum, Cap00, Cap_end,k4)

    # optimze
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_train_step = total_train_step + 1
    if total_train_step % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_reconcile", loss.item(), total_train_step)
writer.close()
model_reconcile.eval()
# 预测
Test_inputs = Test_inputs.to(device)

with torch.no_grad():
    Test_outputs_reconcile = model_reconcile(Test_inputs)
    Test_outputs_reconcile = Test_outputs_reconcile.to(device0)
    test_outputs0_reconcile = np.array(Test_outputs_reconcile).reshape(np.size(test_targets_reconcile, axis=0),
                                                  np.size(test_targets_reconcile, axis=1) - 1)
test_outputs_reconcile=np.zeros((np.size(test_targets_reconcile, axis=0),
                      np.size(test_targets_reconcile, axis=1) - 1))
for i in range(np.size(train_inputs_reconcile,axis=1)):
    test_targets_reconcile[:, i] = test_targets_reconcile[:, i] * Cap[i]
    if i == np.size(train_inputs_reconcile,axis=1)-1:
        break
    else:
        test_outputs_reconcile[:, i] = test_outputs0_reconcile[:, i] *  Cap[i]
test_constrain_reconcile = test_constrain_reconcile * sum(Cap)
test_outputs_cons=test_constrain_reconcile - (np.sum(test_outputs_reconcile,axis=1).reshape(np.size(test_targets_reconcile,axis=0), 1))
test_outputs_reconcile = np.concatenate((test_outputs_reconcile, test_outputs_cons), axis=1).reshape(np.size(test_targets_reconcile,axis=0), np.size(test_targets_reconcile,axis=1))

# calculate rmse
rmse_test__reconcileResult=(np.mean((test_targets_reconcile-test_outputs_reconcile)**2,axis=0))**0.5/Cap
rmse_test_diffO2C_reconcile=rmse_test_OrigResult-rmse_test__reconcileResult
#
train_inputs_reconcile=train_inputs_reconcile*(Cap.reshape([1,np.size(Cap)]))
train_targets_reconcile=train_targets_reconcile*(Cap.reshape([1,np.size(Cap)]))
train_constrain_reconcile=train_constrain_reconcile*sum(Cap)
with torch.no_grad():
    Train_outputs0 = Train_outputs.to(device0)
    train_outputs_reconcile=np.array(Train_outputs0)
train_outputs_reconcile=train_outputs_reconcile*(Cap[0:-1].reshape([1,np.size(Cap)-1]))
train_outputs_cons=train_constrain_reconcile - (np.sum(train_outputs_reconcile,axis=1).reshape(np.size(train_targets_reconcile,axis=0), 1))
train_outputs_reconcile = np.concatenate((train_outputs_reconcile, train_outputs_cons), axis=1).reshape(np.size(train_targets,axis=0), np.size(train_targets,axis=1))
rmse_train_reconcileResult=(np.mean((train_targets_reconcile-train_outputs_reconcile)**2,axis=0))**0.5/Cap
rmse_train_diffO2C_reconcile=rmse_train_OrigResult-rmse_train_reconcileResult
a_reconcile=np.mean(rmse_test_diffO2C_reconcile)
b_reconcile=np.mean(rmse_train_diffO2C_reconcile)

loss1 = loss_fn(Train_inputs[:,1:-1], Train_targets[:, 0:-1], Train_Constrain, Train_targets[:, -1].reshape([Train_targets.size(dim=0),1]), Cap_sum, Cap00, Cap_end,k1)
loss2 = loss_fn(Train_inputs[:,1:-1], Train_targets[:, 0:-1], Train_Constrain, Train_targets[:, -1].reshape([Train_targets.size(dim=0),1]), Cap_sum, Cap00, Cap_end,k2)
loss3 = loss_fn(Train_inputs[:,1:-1], Train_targets[:, 0:-1], Train_Constrain, Train_targets[:, -1].reshape([Train_targets.size(dim=0),1]), Cap_sum, Cap00, Cap_end,k3)
loss4 = loss_fn(Train_inputs[:,1:-1], Train_targets[:, 0:-1], Train_Constrain, Train_targets[:, -1].reshape([Train_targets.size(dim=0),1]), Cap_sum, Cap00, Cap_end,k4)





