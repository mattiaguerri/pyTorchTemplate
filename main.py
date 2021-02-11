'''
Simulate the deployment.
Choose a starting timestamp.
Train, 7 days gap, then test on 7 days.
Then add 7 days to the train set and push the test set 7 days forward.
Re-train and re-test.
Iterate untill all the datset is consumed.
'''


from datetime import timedelta
import LynceusUtils as lu
from models import ffwReg
import numpy as np
import os
import pandas as pd
import pickle
from Project import Project
import sys
from utils import getFeatureImportance, getValidFrames
from utils import processCSVFile
from utils import testNetwork
from trainer import trainer


if __name__ == '__main__':

    project = Project()
    timeKey = project.timeKey
    networkParams = project.networkParams

    if project.valSetOption == 1:
        project.crossValidFolds = 2
        print('\n WARNING: crossValidFolds set to 2')
    elif project.valSetOption == 2:
        project.crossValidFolds = 1
        print('\n WARNING: crossValidFolds set to 1')

    # # build the input dictionary or read it

    fullPath = project.inpDictDir / project.inpDictName
    if os.path.exists(fullPath)==False:
        inpDict = processCSVFile(project.inpCSV, timeKey)
        with open(fullPath, 'wb') as file:
            pickle.dump(inpDict, file)
    else:
        with open(fullPath, 'rb') as file:
            inpDict = pickle.load(file)
    
    # focus only on a specific set of keys
    inpDict['TargetsKeys'] = project.specificTargets

    # dataframe with all the samples
    fullDF = inpDict['DataFrame']

    # init variables
    ts0 = pd.to_datetime(project.testSetStart)
    predsLst = []
    targsLst = []
    timeStampsLst = []

    # loop untill all the test set is consumed

    for ite in range(project.maxIte):

        # get left and right bounds for the test set
        tsL = ts0 + timedelta(days=7)
        tsR = tsL + timedelta(days=7)
        
        print('\n---------------------------------------------')
        print('\nTime Boundaries:')
        print(ts0)
        print(tsL)
        print(tsR)

        # break if test test set is fully consumed
        if tsL > fullDF[timeKey].max():
            break

        # build the test df
        testDF = fullDF[(fullDF[timeKey] >= tsL) & (fullDF[timeKey] < tsR)]

        # check if the test set is empty
        if testDF.shape[0] == 0:
            ts0 = tsL  # move ts0 forward
            continue

        # dataframe with train + valid samples
        fullTrainDF = fullDF[fullDF[timeKey] < ts0]

        # create couples of train and valid sets
        trainValidCouples = getValidFrames(
            fullTrainDF,
            project.crossValidFolds,
            timeKey,
            opt=project.valSetOption)
        
        # train and test the models, loop on the cross validation folds

        crossValidPreds = []

        for fold in range(project.crossValidFolds):

            # train

            # retrieve train and val sets
            trainDF = trainValidCouples[fold][0]
            valDF = trainValidCouples[fold][1]

            print('\nTraining the model, iteration =', ite + 1, ' CV =', fold+1)
            print('Train set size = ', trainDF.shape[0])
            print('Val set size = ', valDF.shape[0])
            print('Test set size = ', testDF.shape[0])

            # scale the dataframes, do not overwrite testDF
            trainDF, valDF, scaTestDF, tarAvg, tarStd = lu.misc.scaleSets(
                trainDF, valDF, testDF,
                inpDict['FeaturesKeys'], inpDict['TargetsKeys']
                )

            # build a train dictionary
            trainDict = {
                'TrainDF' : trainDF,
                'ValDF' : valDF,
                'FeaturesKeys' : inpDict['FeaturesKeys'],
                'TargetsKeys' : inpDict['TargetsKeys']
            }

            # get 20 most imp features
            fulFeaMod = ffwReg(
                n_inp = len(trainDict['FeaturesKeys']),
                n_out = len(trainDict['TargetsKeys']),
                n_hidd = networkParams['n_hidd'],
                n_hidd_layers = networkParams['n_hidd_layers'],
                act_fun = networkParams['act_func'],
                dropout = networkParams['dropout']
                )
            fulFeaMod = trainer(
                fulFeaMod, trainDict, networkParams,
                gpuId=project.setGPU
                )
            featuresImpLst = getFeatureImportance(
                fulFeaMod, trainDict['ValDF'],
                trainDict['FeaturesKeys'], trainDict['TargetsKeys']
                )
            trainDict['FeaturesKeys'] = featuresImpLst[0:20]

            # train the model
            model = ffwReg(
                n_inp = len(trainDict['FeaturesKeys']),
                n_out = len(trainDict['TargetsKeys']),
                n_hidd = networkParams['n_hidd'],
                n_hidd_layers = networkParams['n_hidd_layers'],
                act_fun = networkParams['act_func'],
                dropout = networkParams['dropout']
                )
            model = trainer(
                model, trainDict, networkParams,
                gpuId=project.setGPU,
                verbose=True
                )

            print('\n   Testing the model, iteration =', ite + 1, ' CV =', fold+1)

            # build a test dictionary
            testDict = {
                'TestDF' : scaTestDF,
                'FeaturesKeys' : trainDict['FeaturesKeys'],
                'TargetsKeys' : trainDict['TargetsKeys'],
                'TargetsAvg' : tarAvg,
                'TargetsStd' : tarStd
                }
            
            # test the model
            preds, targs = testNetwork(testDict, model)

            # collect the predictions
            crossValidPreds.append(preds)
        
        # average the predictions of all the CV models
        preds = np.hstack(crossValidPreds)
        preds = np.mean(preds, axis=1)
    
        # collect predictions, targets and timestamps
        predsLst.append(preds)
        targsLst.append(targs)
        timeStampsLst.append(testDF[timeKey].values)
        
        # output results 
        outDict = {'Predictions' : predsLst,
                   'Targets' : targsLst,
                   'TimeStamps' : timeStampsLst
                  }
        outName = project.outDictDir / 'results.pkl'
        with open(outName, 'wb') as file:
            pickle.dump(outDict, file)

        # move ts0 forward
        ts0 = tsL
