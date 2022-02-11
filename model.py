import torch.nn as nn
import torch
from tqdm import tqdm #loading progress bar
import torch.optim as optim
from utils import(check_accuracy, save_checkpoint)
    
learning_rate=1e-4
Device='cuda' if torch.cuda.is_available() else 'cpu'
Batch_size=16
num_epochs=3
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_class): #hidden size_1= 64, xx_2=128, xx_3=64
        super(MLP,self).__init__()
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
        return 


def main():
    model=MLP().to(Device)
    loss_function=nn.BCEWithLogitsLoss()# bianry cross entropy, sigmoid is included in the loss function
                                         # use CROSS entropy loss for multiple classification

    optimizer= optim.Adam(model.parameters(),lr=learning_rate)
              
    check_accuracy(val_loader,model,device=Device)
        
    scaler=torch.cuda.amp.GradScaler()
    
  
    for epoch in range(num_epochs):
        losses=[]
        loop=tqdm(enumerate(train_loader),total=len(train_loader),leave=True)    
        for data in trainset:
            data= data.to(device=Device)# data is the input image
            x,y = data
            #model.zero_grad()
            #forward
            with torch.cuda.amp.autocast():
                predictions=model(data)
                loss=loss_function(predictions , y)
                  
            #backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            #update tadm loop
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            loop.set_postfix(loss=loss.item())
            
        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
        
        #save model
        checkpoint={'state_dict': model.state_dict(),'optimizer':optimizer.state_dict(),}
        save_checkpoint(checkpoint)
        
        #check accuracy
        check_accuracy(val_loader, model,device=Device)
        
    
   
    print('Finished Training')  
    
        
if __name__=='__main__':
    main()