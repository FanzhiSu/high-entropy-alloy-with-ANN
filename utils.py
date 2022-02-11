import torch


def check_accuracy(loader ,model, device='cuda'):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in trainset:
            data= data.to(device=Device)# data is the input image
            x , y = data
            predictions= model(x)
                for idx , i in enumerate (predictions):
                    if torch.argmax(i) == y[i]:
                        correct += 1
                    total += 1
    print('accuracy: ', round (correct/total,3))
    model.train()
    
def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state,filename)

    
                
    