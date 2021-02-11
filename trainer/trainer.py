import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# custom pytorch dataset class
class SputterDataset(Dataset):
    
    def __init__(self, dataInp, dataOut):

        self.dataInp = dataInp
        self.dataOut = dataOut

    def __len__(self):
        
        return self.dataInp.shape[0]

    def __getitem__(self, idx):
        return np.float32(self.dataInp[idx,:]), np.float32(self.dataOut[idx, :])


def trainer(model, trainDict, params, gpuId='cuda:0', verbose=False):

    trainDF = trainDict['TrainDF']
    valDF = trainDict['ValDF']
    feaKey = trainDict['FeaturesKeys']
    tarKey = trainDict['TargetsKeys']

    XTrain = trainDF[feaKey].to_numpy()
    yTrain = trainDF[tarKey].to_numpy()

    XVal = valDF[feaKey].to_numpy()
    yVal = valDF[tarKey].to_numpy()

    etchTrain = SputterDataset(XTrain, yTrain)
    etchTrain = DataLoader(etchTrain,
                           batch_size=params['batch_size'],
                           shuffle=True,
                           num_workers=1)
    
    etchValid = SputterDataset(XVal, yVal)
    etchValid = DataLoader(etchValid,
                           batch_size=params['batch_size'],
                           shuffle=False,
                           num_workers=1)

    device = torch.device(gpuId if torch.cuda.is_available() else "cpu")
    # print("\nUsing the device:", device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=params['lr'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay']
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=50,
        verbose=True,
        factor=0.5
        )

    # train the model, evaluate on validation set

    best_val = np.inf
        
    for epoch in range(params['num_epc']):
            
        # train model
        model.train()
        cnt_batch = 0
        running_loss_train = 0
        for X, y in etchTrain:
            cnt_batch += 1
            X, y  = Variable(X.to(device)), Variable(y.to(device))
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss_train += criterion(out, y).mean()
        running_loss_train /= float(cnt_batch)
    
        scheduler.step(running_loss_train)

        # validate model
        model.eval()
        numSamTot = .0
        running_loss_valid = 0
        for X, y in etchValid:
            X, y = Variable(X.to(device)), Variable(y.to(device))
            numSam = X.shape[0]  # num sample in this batch
            numSamTot += numSam
            out = model(X)
            loss = criterion(out, y)
            running_loss_valid += loss.mean() * numSam
        running_loss_valid /= numSamTot
        
        # store minimum val loss
        if running_loss_valid < best_val:
            best_val = running_loss_valid
        
        if verbose:
            if (epoch+1)%100 == 0:
                print('Epoch:', epoch+1,
                      'Train Loss =', np.round(running_loss_train.cpu().detach().item(), 4),
                      'Valid Loss =', np.round(running_loss_valid.item(), 4))

    model.to('cpu')

    return model
