# functions to be moved in the library

from datetime import timedelta
import LynceusUtils as lu
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import sys
import torch


# set seed for numpy (used also by scikit)
np.random.seed(1)


# add features from n nearest neighbors (in the feature space)
def addNeighborFeature(dataFrames, numNei, feaKey, tarKey, timeKey, addTarget=None):
    """
    Add features from n neighbors in the features space.
    Optionallly, also add a target from the neighbors.
    
    Parameters
    ----------
    dataFrames : list of pandas dataframes
        Order must be train, val, test.
    numNei : integer
        Take features from numNei neighbors
    feaKey : list of strings
        Features keys.
    tarKey : list of strings
        Targets keys.
    timeKey : string
        Key for the time column.
    addTarget : None or string
        If not None, add also this key from the neighs.
    
    Returns
    -------
    outFrames : list of pandas dataframes
        list with train, val and test frames with additional features.
    """

    # build list with keys for output dataframe
    lstOutCols = feaKey + tarKey + ['Time']
    # loop on the number of neigh
    for i in range(numNei):
        # add feats from neigh
        for l in range(len(feaKey)):
            key = feaKey[l] + '_Aug' + str(i+1)
            lstOutCols += [key]
        # add target from neigh
        if addTarget:
            key = addTarget + '_Aug' + str(i+1)
            lstOutCols += [key]
    
    # loop on the three frames
    
    outFrames = []
    
    for dfTmp in dataFrames:
        
        # loop on the rows of the frame
        
        outData = []
        
        for i in range(dfTmp.shape[0]):
        # for i in range(10):
            
            # get indexes of n closer wafers
            vec = dfTmp[feaKey].iloc[i].values  # feats from ith wafer
            X = np.array(dataFrames[0][feaKey])  # look for nei in the train set
            XDif = np.abs(X - vec)
            XL2 = np.linalg.norm(XDif, axis=1)
            inds = XL2.argsort()
            if np.all((XDif[inds[0], :] == .0)):  # remove itself
                inds = inds[1:]
            inds = inds[:numNei]  # get only n nearest neighbors
        
            # build output row using features from the neighbors
            tmpLst = feaKey + [addTarget]
            rowNei = []
            for ind in inds:
                vals = dataFrames[0][tmpLst].iloc[ind].values
                rowNei += list(vals)
        
            # build out row using cols from ith sample
            tmpLst = feaKey + tarKey + [timeKey]
            rowIth = list(dfTmp[tmpLst].iloc[i].values)
            
            # get out row
            row = rowIth + rowNei
            
            outData.append(row)
        
        outFrames.append(pd.DataFrame(data=outData, columns=lstOutCols))
    
    return outFrames


# function to get a dictionary containing sets to be used as input for the modelling
def getInputSets(dfInp, feaKey, tarKey, timeKey,
                 byLot=False,
                 ranges=None, trim=None,
                 augmentData=None, addTarget=None):
    """ 
    Split a dataframe using different criteria.
    By default it does a random split.
    Optional are split by lot and by time windows.
    
    Parameters
    ----------
    dfInp : pandas dataframe
        Dataframe that will be split.
    feaKey : list
       list of features keys
    tarKey : list
       list of targets keys
    timeKey : string
       key of the timestamps column.
    byLot : bool
       Whether top split also by lot or not.
    ranges : None or list
       If not None, ranges foe the test sets.
    trim : None or integer
       If not None, days to trim out of the test set
    augmentData = None or integer
       If not None, number of neighs in features space to add.
    addTarget = None or string
       If not None, a target from the neihs to add as a feature.
    
    Returns
    -------
    outDic : dictionary
        Dictionary of dictionaries, each containing train, val and test df
    """
    
    # get train, val and test sets
    dfTmp = pd.DataFrame.copy(dfInp)
    trainDF, dfValTest = train_test_split(dfTmp, test_size=0.2, random_state=1)
    dfVal, testDF = train_test_split(dfValTest, test_size=0.5, random_state=1)
    
    # eliminate all that is not fea, tar or time
    tmpLst = [timeKey] + tarKey + feaKey
    trainDF = trainDF[tmpLst]
    dfVal = dfVal[tmpLst]
    testDF = testDF[tmpLst]
    
    # scale the sets
    trainDF, dfVal, testDF, tarAvg, tarStd = lu.misc.scaleSets(trainDF,
                                                               dfVal,
                                                               testDF,
                                                               feaKey,
                                                               tarKey)
    
    # augment sets with n nearest neighbors
    if augmentData != None:
        trainDF, dfVal, testDF = addNeighborFeature([trainDF, dfVal, testDF],
                                                    augmentData,
                                                    feaKey,
                                                    tarKey,
                                                    timeKey,
                                                    addTarget)
        feaKeyOut = list(set(trainDF.keys().to_list()) - set(tarKey))
        feaKeyOut.remove(timeKey)
    else:
        feaKeyOut = feaKey
    
    # sort frames by timeKey
    trainDF = trainDF.sort_values(by=[timeKey])
    trainDF = trainDF.reset_index(drop=True)
    dfVal = dfVal.sort_values(by=[timeKey])
    dfVal = dfVal.reset_index(drop=True)
    testDF = testDF.sort_values(by=[timeKey])
    testDF = testDF.reset_index(drop=True)
    
    # collect sets in a dictionary
    timeFrame0 = {'TrainSet' : trainDF,
                  'ValSet' : dfVal,
                  'TestSet' : testDF,
                  'FeaturesKeys' : feaKeyOut,
                  'TargetsKeys' : tarKey,
                  'TargetsAvg' : tarAvg,
                  'TargetsStd' : tarStd}
    
    outDic = {'TimeFrame0' : timeFrame0}
    
    # debug
    print("\nTime Frame 0:")
    print('Train Set =', trainDF.shape)
    print('Val Set =', dfVal.shape)
    print('Test Set = ', testDF.shape)
    print('Total = ', trainDF.shape[0] + dfVal.shape[0] + testDF.shape[0])
    
    # split data keeping into account the lot
    
    if byLot:
        
        # get lists of lots for train, val and test sets
        lstLot = dfInp.Lot.unique()
        trainLot, valTestLot = train_test_split(lstLot, test_size=.2, random_state=1)
        valLot, testLot = train_test_split(valTestLot, test_size=.5, random_state=1)
        
        # get train, val and test sets
        trainDF = dfTmp[dfTmp.Lot.isin(trainLot)]
        dfVal = dfTmp[dfTmp.Lot.isin(valLot)]
        testDF = dfTmp[dfTmp.Lot.isin(testLot)]
        
        # scale the sets
        trainDF, dfVal, testDF, tarAvg, tarStd = lu.misc.scaleSets(trainDF, dfVal, testDF,
                                                                   feaKey, tarKey)
        
        # sort frames by timeKey
        trainDF = trainDF.sort_values(by=[timeKey])
        trainDF = trainDF.reset_index(drop=True)
        dfVal = dfVal.sort_values(by=[timeKey])
        dfVal = dfVal.reset_index(drop=True)
        testDF = testDF.sort_values(by=[timeKey])
        testDF = testDF.reset_index(drop=True)
        
        # collect sets in a dictionary
        lotSet = {'TrainSet' : trainDF,
                  'ValSet' : dfVal,
                  'TestSet' : testDF,
                  'FeaturesKeys' : feaKey,
                  'TargetsKeys' : tarKey,
                  'TargetsAvg' : tarAvg,
                  'TargetsStd' : tarStd}
        
        outDic['SplitByLot'] = lotSet
        
        # debug
        print("\nSplitting by Lot Sets:")
        print(trainDF.shape)
        print(dfVal.shape)
        print(testDF.shape)
    
    # get the sets split by time interval
    
    if ranges:
        ranNum = 0  # number id for the range
        for ran in ranges:
            ranNum += 1
            
            # get test dataframe
            if isinstance(ran[1], int):
                ts0 = pd.to_datetime(ran[0])
                ts1 = ts0 + timedelta(days=ran[1])
            else:
                ts0 = pd.to_datetime(ran[0])
                ts1 = pd.to_datetime(ran[1])
            testDF = pd.DataFrame.copy(dfInp[(dfInp[timeKey] >= ts0) & (dfInp[timeKey] < ts1)])

            # get train and validation sets, I
            # get train + val dataframe
            trainDFVal = pd.concat([dfInp, testDF])
            trainDFVal.drop_duplicates(keep=False, inplace=True)
            # split train and val dataframes    
            trainDF, dfVal = train_test_split(trainDFVal, test_size=.1, random_state=1)

            # # get train and validation sets, II
            # # alternative way to get the validation set,
            # # take values only close to the test set
            # numDay = 1
            # # loop increasing the numDay untill the
            # # size of val set is >= 5% size train set
            # for count in range(30*4):  # 4 months each side max
            #     tsL = ts0 - timedelta(days=numDay)  # left limit val set 
            #     tsR = ts1 + timedelta(days=numDay)  # right limit val set
            #     dfValL = pd.DataFrame.copy(dfInp[(dfInp[timeKey] >= tsL) & (dfInp[timeKey] < ts0)])
            #     dfValR = pd.DataFrame.copy(dfInp[(dfInp[timeKey] >= ts1) & (dfInp[timeKey] < tsR)])
            #     dfVal = pd.concat([dfValL, dfValR])
            #     trainDF = pd.concat([dfInp, dfVal, testDF]).drop_duplicates(keep=False)
            #     # break if size of val set reaches at least 5% size of train set
            #     val = trainDF.shape[0]//100*5
            #     if dfVal.shape[0] >= val:
            #         break
            #     else:
            #         numDay += 1

            # eliminate all that is not fea, tar or time
            tmpLst = [timeKey] + tarKey + feaKey
            trainDF = trainDF[tmpLst]
            dfVal = dfVal[tmpLst]
            testDF = testDF[tmpLst]
            
            # scale the sets
            trainDF, dfVal, testDF, tarAvg, tarStd = lu.misc.scaleSets(trainDF, dfVal, testDF,
                                                                       feaKey, tarKey)
            
            # augment sets with n nearest neighbors
            if augmentData != None:
                trainDF, dfVal, testDF = augmentDataFrame([trainDF, dfVal, testDF],
                                                          augmentData,
                                                          feaKey,
                                                          tarKey,
                                                          timeKey,
                                                          addTarget)
                feaKeyOut = list(set(trainDF.keys().to_list()) - set(tarKey))
                feaKeyOut.remove(timeKey)
            else:
                feaKeyOut = feaKey
           
            # sort frames by timeKey
            trainDF = trainDF.sort_values(by=[timeKey])
            trainDF = trainDF.reset_index(drop=True)
            dfVal = dfVal.sort_values(by=[timeKey])
            dfVal = dfVal.reset_index(drop=True)
            testDF = testDF.sort_values(by=[timeKey])
            testDF = testDF.reset_index(drop=True)
            
            # trim the test set
            if trim:
                ts0Trim = pd.to_datetime(ts0 + timedelta(days=trim))
                ts1Trim = pd.to_datetime(ts1 - timedelta(days=trim))
                testDFTrim = pd.DataFrame.copy(testDF[(testDF[timeKey] >= ts0Trim) & (testDF[timeKey] < ts1Trim)])
                testDF = pd.DataFrame.copy(testDFTrim)
                testDF = testDF.sort_values(by=[timeKey])
                testDF = testDF.reset_index(drop=True)
            
            # collect sets in a dictionary
            timeFrameSet = {'TrainSet' : trainDF,
                            'ValSet' : dfVal,
                            'TestSet' : testDF,
                            'FeaturesKeys' : feaKeyOut,
                            'TargetsKeys' : tarKey,
                            'TargetsAvg' : tarAvg,
                            'TargetsStd' : tarStd}
            
            outDic['TimeFrame'+str(ranNum)] = timeFrameSet
            
            # debug
            print("\nTime Frame " + str(ranNum) + ":")
            print('Train Set =', trainDF.shape)
            print('Val Set =', dfVal.shape)
            print('Test Set = ', testDF.shape)
            print('Total = ', trainDF.shape[0] + dfVal.shape[0] + testDF.shape[0])
            
    return outDic


# process the csv file and obtain a dictionary with the data
def processCSVFile(csvFile, timeKey):
    
    # read the data
    dfOriginal = pd.read_csv(csvFile)
    print('\nShape of original dataframe = ', dfOriginal.shape)

    # drop features related to adjacent samples
    dfTmp = pd.DataFrame.copy(dfOriginal)
    lstAng = [k for k in dfTmp.keys() if '_angle_' in k]  # list all ang models
    lstKeyFull = dfTmp.keys().to_list()
    keyToRem = []
    for num in range(15):
        st = '_' + str(num)
        for k in lstKeyFull:
            if k[-len(st):] == st and k not in lstAng:
                keyToRem.append(k)
    keyToKeep = list(set(lstKeyFull) - set(keyToRem))
    keyToKeep = [k for k in keyToKeep if 'DeltaTime' not in k]
    keyToKeep = sorted(keyToKeep)
    dfOneSample = pd.DataFrame.copy(dfTmp[keyToKeep])
    print('\nNumber of cols after removing features from neigh =', dfOneSample.shape[1])

    # get a list of all the angle models
    lstAng = [x for x in dfOneSample.keys() if '_angle_' in x]
    print('\nList of all the angle models:')
    _ = [print(x) for x in lstAng]

    # get a df copy and sort by time
    dfFull = dfOneSample.copy()
    dfFull.Time = pd.to_datetime(dfFull.Time)
    dfFull.sort_values(by='Time', inplace=True)
    dfFull = dfFull.reset_index(drop=True)

    # check for flat cols and drop them
    dfTmp = pd.DataFrame.copy(dfFull)
    flatCols = []
    print('\nList of flat cols:')
    for key in dfTmp.keys():
        if dfTmp[key].min() == dfTmp[key].max() and dfTmp[key].std()==0:
            flatCols.append(key)
            print(key)
    dfTmp = dfTmp.drop(flatCols, axis=1)
    print('\nNumber of cols before dropping:', dfFull.shape[1])
    print('Number of cols after dropping:', dfTmp.shape[1])
    dfNotFlat = pd.DataFrame.copy(dfTmp)

    # remove a list of selected columns
    keysToRem = None
    if keysToRem:
        dfSel = dfNotFlat.drop(keysToRem, axis=1)
    else:
        dfSel = pd.DataFrame.copy(dfNotFlat)

    # get the list of the features
    feaKey = dfSel.keys().to_list()
    feaKey.remove(timeKey)
    if 'Comb' in dfSel.keys():
        feaKey.remove('Comb')
    if 'Status' in dfSel.keys():
        feaKey.remove('Status')
    [feaKey.remove(k) for k in lstAng]

    # get the list of targets
    # list all ang models
    lstAng = [k for k in dfSel.keys() if '_angle_' in k]
    tarKey = lstAng

    # check
    print('\nTotal number of keys:', len(dfSel.keys()))
    print('\nNumber of features:', len(feaKey))
    print('Number of targets:', len(tarKey))
    print('\nList of targets:')
    _ = [print(x) for x in tarKey]

    # sort by time
    dfSel.sort_values(by='Time', inplace=True)
    dfSel.reset_index(drop=True, inplace=True)

    # build output dictionary
    inputDF = pd.DataFrame.copy(dfSel)
    outDict = {'DataFrame' : inputDF,
               'FeaturesKeys' : feaKey,
               'TargetsKeys' : tarKey
               }

    return outDict


# test a NN and print two metrics
def testNetwork(testDict, model):
    
    model.eval()
    
    feaKey = testDict['FeaturesKeys']
    tarKey = testDict['TargetsKeys']
    XTest = testDict['TestDF'][feaKey].to_numpy()
    yTest = testDict['TestDF'][tarKey].to_numpy()

    XTest = torch.from_numpy(XTest).float()
    yTest = torch.from_numpy(yTest).float()
    yHat = model(XTest).detach()

    # rescale predictions and targets
    tarAvg = testDict['TargetsAvg']
    tarStd = testDict['TargetsStd']
    yHat = yHat * tarStd + tarAvg
    yTest = yTest * tarStd + tarAvg

    # get metrics
    mae = metrics.mean_absolute_error(yTest, yHat)
    r2 = metrics.r2_score(yTest, yHat)

    print('   MAE =', mae)
    print('   R2 =', r2)

    return yHat, yTest


# wrap a model into a scikit estimator
class Wrapper(BaseEstimator, RegressorMixin):

    def __init__(self, model, **model_hyper_parameters):
        """
        """
        super().__init__()
        self.model = model

    def fit(self, X, Y=None):
        """
        Fit global model on X features to minimize 
        a given function on Y.

        @param X
        @param Y
        """
        # TODO
        return self

    def predict(self, X):
        """
        @param X: features vector the model will be evaluated on
        """
        # compute predictions
        X = torch.from_numpy(X).float()
        pre = self.model(X).detach().numpy()
        
        return pre


def getFeatureImportance(model, valDF, featuresKeys, targetsKeys, verbose=False):

    XVal = valDF[featuresKeys].to_numpy()
    yVal = valDF[targetsKeys].to_numpy()

    # wrap model in a scikit compatible class
    wrappedModel = Wrapper(model)

    # permutation feature importance
    perImp = permutation_importance(wrappedModel, XVal, yVal,
                                    scoring=metrics.make_scorer(metrics.r2_score),
                                    n_repeats=128,
                                    n_jobs=8,
                                    random_state=0)

    # collect the features in a sorted list
    lstImpFea = []
    for i in perImp.importances_mean.argsort()[::-1]:
        if perImp.importances_mean[i] - 2 * perImp.importances_std[i] > 0:
            lstImpFea.append(featuresKeys[i])
            if verbose:
                print(f"{featuresKeys[i]:<8}"
                      f"{perImp.importances_mean[i]:9.3f}"
                      f" +/- {perImp.importances_std[i]:.3f}")
    
    return lstImpFea


# get couples of train and valid sets
def getValidFrames(fullTrainDF, crossValFolds, timeKey, opt=0):

    # init output
    trainValidCouples = []

    # get the size of the val set
    # # no bigger than 10% of the fullTrainDF
    valSize0 = fullTrainDF.shape[0] // crossValFolds
    valSize1 = fullTrainDF.shape[0] // 10
    valSize = np.min([valSize0, valSize1])

    # random split, consider all the samples
    if opt==0:
        
        # shuffle the dataframe
        fullTrainDF = fullTrainDF.sample(frac=1, random_state=1)
        
        # get indexes to define the validation sets
        lstInd = []
        for i in range(crossValFolds):
            indL = i * valSize
            indR = indL + valSize
            lstInd.append([indL, indR])

        # get the dataframes couples
        for indVal in range(crossValFolds):
            indL = lstInd[indVal][0]
            indR = lstInd[indVal][1]
            valDF = fullTrainDF[indL:indR]
            tmpDF = pd.concat([fullTrainDF, valDF])
            trainDF = tmpDF.drop_duplicates(keep=False, inplace=False)
            trainValidCouples.append([trainDF, valDF])

    # random split, only consider samples close to the test set
    if opt==1:

        # get a df of size valSize*2
        maxTime = fullTrainDF[timeKey].max()
        valStart = maxTime
        tmpDF = fullTrainDF[(fullTrainDF[timeKey] > valStart) & (fullTrainDF[timeKey] <= maxTime)]
        while tmpDF.shape[0] < valSize*2:
            valStart -= timedelta(days=1)
            tmpDF = fullTrainDF[(fullTrainDF[timeKey] > valStart) & (fullTrainDF[timeKey] <= maxTime)]
        
        # half tmpDF goes into the trainDF
        # the other half is the valDF
        tmpDF_1, tmpDF_2 = train_test_split(tmpDF, test_size=.5)

        # tmpDF_1 is the valDF
        trainDF = pd.concat([fullTrainDF, tmpDF_1])
        trainDF = trainDF.drop_duplicates(keep=False)
        valDF = tmpDF_1
        trainValidCouples.append([trainDF, valDF])

        # tmpDF_2 is the valDF
        trainDF = pd.concat([fullTrainDF, tmpDF_2])
        trainDF = trainDF.drop_duplicates(keep=False)
        valDF = tmpDF_2
        trainValidCouples.append([trainDF, valDF])
    
    # only consider samples close to the test set
    if opt==2:

        maxTime = fullTrainDF[timeKey].max()
        valStart = maxTime
        valDF = fullTrainDF[(fullTrainDF[timeKey] > valStart) & (fullTrainDF[timeKey] <= maxTime)]
        while valDF.shape[0] < valSize:
            valStart -= timedelta(days=1)
            valDF = fullTrainDF[(fullTrainDF[timeKey] > valStart) & (fullTrainDF[timeKey] <= maxTime)]
        
        trainDF = pd.concat([fullTrainDF, valDF])
        trainDF = trainDF.drop_duplicates(keep=False)
        trainValidCouples.append([trainDF, valDF])
    
    return trainValidCouples
