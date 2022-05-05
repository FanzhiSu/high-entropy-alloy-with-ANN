# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:26:34 2022

@author: Clark
"""

import torch
import torch.nn as nn
from tqdm import tqdm #loading progress bar
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
'''  for visualisation delete afterwards '''
from torch.utils.tensorboard import SummaryWriter 
import sys
'''delete till here'''

Writer = SummaryWriter('runs/HEM_1')
learning_rate=1e-3
Device='cuda' if torch.cuda.is_available() else 'cpu'
Batch_size=128
num_epochs=1
Pin_memory= True
val_percent: float = 0.3   


def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state,filename)

    

class HEM_train(Dataset): #high entropy metal
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('C:/Users/clarq/Desktop/input_final.csv', delimiter=' ', dtype=np.float32)
        xy=xy[:90900,:]
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        sc = StandardScaler()
        self.x_data = torch.from_numpy(sc.fit_transform(xy[:, :13])) # size [n_samples, n_features]
        # self.val_x_data = torch.from_numpy(sc.fit_transform(xy[86860:, :13]))
        self.y_data = torch.from_numpy(xy[:, [13]]) # size [n_samples, 1]
        # self.val_y_data =  torch.from_numpy(xy[86860:, [13]])
        # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

        # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class HEM_val(Dataset): #high entropy metal
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy_val = np.loadtxt('C:/Users/clarq/Desktop/input_final.csv', delimiter=' ', dtype=np.float32)
        xy_val=xy_val[90900:,:]
        self.n_samples_val = xy_val.shape[0]

        # here the first column is the class label, the rest are the features
        sc = StandardScaler()
        # self.x_data = torch.from_numpy(sc.fit_transform(xy[:, :13])) # size [n_samples, n_features]
        self.val_x_data = torch.from_numpy(sc.fit_transform(xy_val[:, :13]))
        # self.y_data = torch.from_numpy(xy[:, [13]]) # size [n_samples, 1]
        self.val_y_data =  torch.from_numpy(xy_val[:, [13]])
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.val_x_data[index], self.val_y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples_val


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_class): #hidden size_1= 64, xx_2=128, xx_3=64
        super(MLP,self).__init__()
        self.input_size=input_size
        self.fc1 = nn.Linear(input_size,hidden_size_1)
        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2,hidden_size_3)
        self.relu3=nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_3,num_class)
     
    
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu1(out)
        out=self.fc2(out)
        out=self.relu2(out)
        out=self.fc3(out)
        out=self.relu3(out)
        out=self.fc4(out)
        return  out



def main():
    model=MLP(13,64,128,64,1).to(Device)
    dataset_train = HEM_train()
    dataset_val = HEM_val()

    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val

    # train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val],
    #                                                generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset_train, shuffle=True,batch_size=Batch_size,pin_memory=Pin_memory,num_workers=0)
    val_loader = DataLoader(dataset_val, shuffle=False,batch_size=Batch_size,pin_memory=Pin_memory,num_workers=0)
    loss_function=nn.BCEWithLogitsLoss()# bianry cross entropy, sigmoid is included in the loss function
                                           # use CROSS entropy loss for multiple classification

    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    ''' to check accuracy (delete afterwards)'''
    dataset = iter(train_loader) # visualisation
    data,key = dataset.next()
    data=data.to(device=Device)
    Writer.add_graph(model,data)
    Writer.close()
    # sys.exit()
    '''  delete till here '''

    n_total_steps=len(train_loader)
    running_loss = 0.0
    running_correct = 0.0
    for epoch in range(num_epochs):
        losses=[]
        loop=tqdm(enumerate(train_loader),total=len(train_loader),leave=True)    
        for batch_idx, (data, key) in loop:
            data= data.to(device=Device)# data is the input image
            key= key.to(device=Device)
            #model.zero_grad()
            #forward



            predictions=model(data)
            loss=loss_function(predictions,key)
            losses.append(loss.item())
                  
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _,predicted = torch.max(predictions,1)
            running_correct += (predicted == key).sum().item()
            
            ''' to check accuracy (delete afterwards)'''
            if (batch_idx+1) % 100 == 0:
                Writer.add_scalar ('training loss', running_loss/100, epoch*n_total_steps+batch_idx)
                Writer.add_scalar ('training accuracy', running_correct/100, epoch*n_total_steps+batch_idx)
                running_loss = 0.0
                running_correct = 0.0
                
            '''  delete till here '''
           
        
            #update tadm loop
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            loop.set_postfix()
            
        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
        
        #save model
        checkpoint={'state_dict': model.state_dict(),'optimizer':optimizer.state_dict(),}
        save_checkpoint(checkpoint)
        
        
        # Check accuracy on training & test to see how good our model


        def check_accuracy(loader, model):
            if loader==train_loader:
                print("Checking accuracy on training data")
            else:
                print("Checking accuracy on test data")

            num_correct = 0
            num_samples = 0
            model.eval()
    
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device=Device)
                    y = y.to(device=Device)
                    predictions = torch.sigmoid(model(x))
                    predictions = (predictions>0.5).float()
                    num_correct += (predictions == y).sum()
                    num_samples += predictions.size(0)
            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

                    
                   # _, predictions = scores.max(1)
                   # if predictions == y:
                   #     num_correct = num_correct + 1
                
            model.train()
        
        
        check_accuracy(val_loader, model)
        
    
   
    print('Finished Training')  
    
        
if __name__=='__main__':
    main()