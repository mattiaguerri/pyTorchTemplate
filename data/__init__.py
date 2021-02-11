from .customDataset import CustomDataset
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import DataLoader, random_split

def getDataLoaders(pathToDict, namSet, speAngMod, batchSize):
    """
    This function returns the train, val and test dataloaders.
    """

    # load input dictionary
    with open(pathToDict, 'rb') as f:
        dictSets = pickle.load(f)
    
    dfTrain = dictSets[namSet]['TrainSet']
    dfVal = dictSets[namSet]['ValSet']
    dfTest = dictSets[namSet]['TestSet']
    feaKey = dictSets[namSet]['FeaturesKeys']
    
    tarKey = [speAngMod]
    
    XTrain = dfTrain[feaKey].to_numpy()
    yTrain = dfTrain[tarKey].to_numpy()

    XVal = dfVal[feaKey].to_numpy()
    yVal = dfVal[tarKey].to_numpy()

    XTest = dfTest[feaKey].to_numpy()
    yTest = dfTest[tarKey].to_numpy()

    # build the datasets
    train_ds = CustomDataset(XTrain, yTrain)
    val_ds = CustomDataset(XVal, yVal)
    test_ds = CustomDataset(XTest, yTest)

    train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batchSize, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batchSize, shuffle=False)

    return train_dl, val_dl, test_dl