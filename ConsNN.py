import time
import numpy as np
import torch
from torch import nn, tensor
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio


dataFile = 'data_demo'
fore_hier = scio.loadmat(dataFile)
child_base_forecast_test=fore_hier['child_base_forecast_test']
child_base_forecast_train=fore_hier['child_base_forecast_train']
child_target_test=fore_hier['child_target_test']
child_target_train=fore_hier['child_target_train']
child_cap=fore_hier['child_cap']
parent_target_train=fore_hier['parent_target_train']
parent_target_test=fore_hier['parent_target_test']
parent_base_forecast_train=fore_hier['parent_base_forecast_train']
parent_base_forecast_test=fore_hier['parent_base_forecast_test']
parent_cap=fore_hier['child_cap']
# define device
device=torch.device("cuda")
device0=torch.device("cpu")
# the network include Constraint learning module, Constraint reconciliation module
# data processing（normalize）
test_inputs0=child_base_forecast_test
test_targets0=child_target_test
test_constrain0=parent_base_forecast_test
train_inputs0=child_base_forecast_train
train_targets0=child_target_train
train_constrain0=parent_base_forecast_train
Cap=child_cap
train_targets=np.zeros([np.size(train_targets0,axis=0),np.size(train_targets0,axis=1)])
test_targets=np.zeros([np.size(test_targets0,axis=0),np.size(test_targets0,axis=1)])
train_inputs=np.zeros([np.size(train_inputs0,axis=0),np.size(train_inputs0,axis=1)])
test_inputs=np.zeros([np.size(test_inputs0,axis=0),np.size(test_inputs0,axis=1)])
for i in range(np.size(train_inputs0,axis=1)):
    train_inputs[:,i]=train_inputs0[:,i]/Cap[0,i]
    train_targets[:,i]=train_targets0[:,i]/Cap[0,i]
    test_inputs[:,i]=test_inputs0[:,i]/Cap[0,i]
    test_targets[:,i]=test_targets0[:,i]/Cap[0,i]
train_constrain = train_constrain0/np.sum(Cap)
test_constrain = test_constrain0/np.sum(Cap)
#############################—————————————————————————————————————————————##############################################
#############################———————————Constraint learning module————————##############################################
#############################—————————————————————————————————————————————##############################################
print('#############################———————————         Constraint learning module      —————————##############################################')
class mlp_conslearning(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=128, hidden_size2=64, hidden_size3=32):
        super(mlp_conslearning, self).__init__()
        self.model1 = Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size3),
            nn.Sigmoid(),
            nn.Linear(hidden_size3, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, output_size),
        )

    def forward(self, x):
        y = self.model1(x)
        return y
#
input_size = np.size(train_inputs,axis=1)+1
# hidden_size1 = 40
output_size = np.size(train_targets,axis=1)
model_conslearning = mlp_conslearning(input_size=input_size, output_size=output_size)
model_conslearning = model_conslearning.to(device)
lossMSE = nn.MSELoss()
lossMSE = lossMSE.to(device)
class SelfCorrectingLoss(nn.Module):
    def __init__(self):
        super(SelfCorrectingLoss, self).__init__()
        return

    def forward(self, outputs, targets, train_Constrain,Cap_sum,Cap00,k):
        loss1 = lossMSE(targets, outputs)

        Cons_loss = (train_Constrain*Cap_sum-((torch.sum(outputs*Cap00,dim=1)).reshape(train_Constrain.size(dim=0),1)))/Cap_sum
        loss2 = (Cons_loss ** 2)*k
        loss2 = torch.mean(loss2)
        loss = torch.mean(loss1 + loss2)
        return loss
loss_fn_SelfCorrecting = SelfCorrectingLoss()
loss_fn_SelfCorrecting = loss_fn_SelfCorrecting.to(device)
# optimizer
learning_rate= 1e-1
optimizer = torch.optim.SGD(model_conslearning.parameters(), lr=learning_rate)
# prepare data
Train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
Train_targets = torch.tensor(train_targets, dtype=torch.float32)
Train_Constrain = torch.tensor(train_constrain, dtype=torch.float32)
Test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
Test_Constrain = torch.tensor(test_constrain, dtype=torch.float32)
Test_targets = torch.tensor(test_targets, dtype=torch.float32)
Cap_sum=torch.tensor(np.sum(Cap), dtype=torch.float32)
Cap00=torch.tensor(Cap, dtype=torch.float32)

# train
# Record the step of training
total_train_step = 0
# Record the step of testing
total_test_step = 0
# Number of epoch
epoch_learning = 100
## Relaxation
# epoch
epoch__Relaxation = 40
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
        Train_outputs = model_conslearning(torch.cat((Train_inputs, Train_Constrain), dim=1))
        # loss
        loss = loss_fn_SelfCorrecting(Train_outputs, Train_targets, Train_Constrain, Cap_sum, Cap00, lamb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            lossMSE_train=lossMSE(Train_outputs, Train_targets)
            Test_Constrain = Test_Constrain.to(device)
            Test_inputs = Test_inputs.to(device)
            Test_targets = Test_targets.to(device)
            Test_outputs = model_conslearning(torch.cat((Test_inputs, Test_Constrain), dim=1))
            lossMSE_test = lossMSE(Test_outputs, Test_targets)
            end_time = time.time()
            print(end_time - start_time)
            print("train_step: {}，loss: {}".format(total_train_step, loss.item()))
            print("train_step: {}，lossMSE: {}".format(total_train_step, lossMSE_train.item()))
            writer.add_scalar("train_loss_conslearning", lossMSE_train.item(), total_train_step)
    # update Multiplier
    with torch.no_grad():
        # Using the base forecasting processed by the constraint learning module as feasible solutions for the dual problem
        loss_UP = lossMSE(Train_inputs*Cap00/(torch.sum(Train_inputs*Cap00,dim=1).reshape(Train_outputs.size(dim=0),1))*Train_Constrain*Cap_sum/Cap00, Train_targets)
        loss_lamb = lossMSE(Train_outputs, Train_targets)
        StepSize = 1 / torch.mean(abs((Train_Constrain*Cap_sum-((torch.sum(Train_outputs*Cap00,dim=1)).reshape(Train_Constrain.size(dim=0),1)))/Cap_sum))
        SubGrad= (loss_lamb-loss_UP)
        lamb = lamb + SubGrad*StepSize
        if lamb<=0:
            lamb=0
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
Test_Constrain = Test_Constrain.to(device)
with torch.no_grad():
    Test_outputs_conslearning = model_conslearning(torch.cat((Test_inputs, Test_Constrain), dim=1))
    Test_outputs_conslearning = Test_outputs_conslearning.to(device0)
    test_outputs0_conslearning = np.array(Test_outputs_conslearning).reshape(np.size(test_targets, axis=0),
                                                  np.size(test_targets, axis=1))
test_outputs_conslearning=np.zeros((np.size(test_targets, axis=0),
                      np.size(test_targets, axis=1) ))
for i in range(np.size(test_targets,axis=1)):
    test_targets[:, i] = test_targets[:, i] * Cap[0,i]
    test_outputs_conslearning[:, i] = test_outputs0_conslearning[:, i] * Cap[0,i]
# calculating rmse
rmse_test_ConslearningResult=(np.mean((test_targets-test_outputs_conslearning)**2,axis=0))**0.5/Cap
rmse_test_OrigResult=(np.mean((test_targets-test_inputs0)**2,axis=0))**0.5/Cap
rmse_test_diffO2C_conslearning=rmse_test_OrigResult-rmse_test_ConslearningResult
#
train_targets=train_targets*(Cap.reshape([1,np.size(Cap)]))
train_constrain=train_constrain*np.sum(Cap)
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
#############################———————————Constraint reconciliation module—————————##############################################
#############################—————————————————————————————————————————————##############################################
print('#############################———————————         Constraint reconciliation module      —————————##############################################')

# data processing
train_inputs_reconcile0=train_outputs_conslearning
test_inputs_reconcile0=test_outputs_conslearning

train_targets_reconcile=np.zeros([np.size(train_targets0,axis=0),np.size(train_targets0,axis=1)])
test_targets_reconcile=np.zeros([np.size(test_targets0,axis=0),np.size(test_targets0,axis=1)])
train_inputs_reconcile=np.zeros([np.size(train_inputs0,axis=0),np.size(train_inputs0,axis=1)])
test_inputs_reconcile=np.zeros([np.size(test_inputs0,axis=0),np.size(test_inputs0,axis=1)])
for i in range(np.size(train_inputs_reconcile0,axis=1)):
    train_inputs_reconcile[:,i]=train_inputs_reconcile0[:,i]/Cap[0,i]
    train_targets_reconcile[:,i]=train_targets0[:,i]/Cap[0,i]
    test_inputs_reconcile[:,i]=test_inputs_reconcile0[:,i]/Cap[0,i]
    test_targets_reconcile[:,i]=test_targets0[:,i]/Cap[0,i]
train_constrain_reconcile = train_constrain0/np.sum(Cap)
test_constrain_reconcile = test_constrain0/np.sum(Cap)
class SymmetricActivation(nn.Module):
    def forward(self, z):
        signed = torch.tanh(z)
        sigma_s = signed / torch.sum(signed, dim=1, keepdim=True)  #
        return sigma_s
class RGU(nn.Module):
    def __init__(self, num_children):
        super(RGU, self).__init__()
        self.num_children = num_children
        input_dim = num_children + 1  # n_l + ε
        hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_children)  # Output z_i for each child
        )
        self.symmetric_activation = SymmetricActivation()


    def forward(self, x_cpm, x_cons):
        """
        x_cpm: Tensor of shape (batch_size, n_l)
        x_cons: Tensor of shape (batch_size, 1)
        Returns: reconciled forecast (batch_size, n_l)
        """
        epsilon = (x_cons*Cap_sum - torch.sum(x_cpm*Cap00, dim=1, keepdim=True))/Cap_sum  # (batch_size, 1)
        concat_input = torch.cat([x_cpm, epsilon], dim=1)         # (batch_size, n_l+1)

        z = self.fc(concat_input)                                 # (batch_size, n_l)
        h = self.symmetric_activation(z)                          # (batch_size, n_l)
        x_c = (x_cpm*Cap00 + h * epsilon*Cap_sum)/Cap00                                 # (batch_size, n_l)

        return x_c
class ConstraintReconciliationModule(nn.Module):
    def __init__(self, num_rgu=20, num_children=15):
        super(ConstraintReconciliationModule, self).__init__()
        self.rgus = nn.ModuleList([
            RGU(num_children=num_children) for _ in range(num_rgu)
        ])

    def forward(self, x_cpm, x_cons):
        """
        x_cpm: Tensor of shape (batch_size, 15)
        x_cons: Tensor of shape (batch_size, 1)
        Returns: reconciled forecast of shape (batch_size, 15)
        """
        outputs = []
        for rgu in self.rgus:
            out = rgu(x_cpm, x_cons)        # shape: (batch_size, 15)
            outputs.append(out.unsqueeze(0))

        all_outputs = torch.cat(outputs, dim=0)  # shape: (20, batch_size, 15)
        mean_output = torch.mean(all_outputs, dim=0)  # shape: (batch_size, 15)
        return mean_output
model_CRM = ConstraintReconciliationModule(num_rgu=64, num_children=np.size(train_targets,axis=1))
model_CRM = model_CRM.to(device)
lossMSE = nn.MSELoss()
lossMSE = lossMSE.to(device)
# optimizer
learning_rate= 1e-2
optimizer = torch.optim.SGD(model_CRM.parameters(), lr=learning_rate)
# data prepare
Train_inputs = torch.tensor(train_inputs_reconcile, dtype=torch.float32)
Train_targets = torch.tensor(train_targets_reconcile, dtype=torch.float32)
Train_Constrain = torch.tensor(train_constrain_reconcile, dtype=torch.float32)
Test_inputs = torch.tensor(test_inputs_reconcile, dtype=torch.float32)
Test_targets = torch.tensor(test_targets_reconcile, dtype=torch.float32)
Test_Constrain = torch.tensor(test_constrain_reconcile, dtype=torch.float32)
Cap_sum=torch.tensor(np.sum(Cap), dtype=torch.float32)
Cap00=torch.tensor(Cap, dtype=torch.float32)
# train
# Record the step of training
total_train_step = total_train_step
# training epoch
epoch1 = 10000
# add tensorboard
start_time = time.time()
for i in range(epoch1):
    # start training
    model_CRM.train()
    Train_Constrain = Train_Constrain.to(device)
    Train_inputs = Train_inputs.to(device)
    Train_targets = Train_targets.to(device)
    Cap_sum = Cap_sum.to(device)
    Cap00 = Cap00.to(device)
    # for data in train_inputs:
    Train_outputs = model_CRM(Train_inputs,Train_Constrain)
    # loss function
    loss=lossMSE(Train_outputs, Train_targets)
    # optimze
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_train_step = total_train_step + 1
    if total_train_step % 100 == 0:
        end_time = time.time()
        Test_Constrain = Test_Constrain.to(device)
        Test_inputs = Test_inputs.to(device)
        Test_targets = Test_targets.to(device)
        Test_outputs = model_CRM(Test_inputs, Test_Constrain)
        lossMSE_test = lossMSE(Test_outputs, Test_targets)
        print(end_time - start_time)
        print("train_step: {}，lossMSE: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss_conslearning", loss.item(), total_train_step)
writer.close()
model_CRM.eval()
# test
Test_inputs = Test_inputs.to(device)
with torch.no_grad():
    Test_Constrain = Test_Constrain.to(device)
    Test_outputs_reconcile = model_CRM(Test_inputs, Test_Constrain)
    Test_outputs_reconcile = Test_outputs_reconcile.to(device0)
    test_outputs0_reconcile = np.array(Test_outputs_reconcile).reshape(np.size(test_targets_reconcile, axis=0),
                                                  np.size(test_targets_reconcile, axis=1))
test_outputs_reconcile=np.zeros((np.size(test_targets_reconcile, axis=0),
                      np.size(test_targets_reconcile, axis=1)))
for i in range(np.size(train_inputs_reconcile,axis=1)):
    test_targets_reconcile[:, i] = test_targets_reconcile[:, i] * Cap[0,i]
    test_outputs_reconcile[:, i] = test_outputs0_reconcile[:, i] *  Cap[0,i]
test_constrain_reconcile = test_constrain_reconcile * np.sum(Cap)
# calculate rmse
rmse_test__reconcileResult=(np.mean((test_targets_reconcile-test_outputs_reconcile)**2,axis=0))**0.5/Cap
rmse_test_diffO2C_reconcile=rmse_test_OrigResult-rmse_test__reconcileResult
#
train_inputs_reconcile=train_inputs_reconcile*(Cap.reshape([1,np.size(Cap)]))
train_targets_reconcile=train_targets_reconcile*(Cap.reshape([1,np.size(Cap)]))
train_constrain_reconcile=train_constrain_reconcile*np.sum(Cap)
with torch.no_grad():
    Train_outputs0 = Train_outputs.to(device0)
    train_outputs_reconcile=np.array(Train_outputs0)
train_outputs_reconcile=train_outputs_reconcile*Cap
rmse_train_reconcileResult=(np.mean((train_targets_reconcile-train_outputs_reconcile)**2,axis=0))**0.5/Cap
rmse_train_diffO2C_reconcile=rmse_train_OrigResult-rmse_train_reconcileResult
a_reconcile=np.mean(rmse_test_diffO2C_reconcile)
b_reconcile=np.mean(rmse_train_diffO2C_reconcile)

