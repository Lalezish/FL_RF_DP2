# Author: Alexander Frid
# Date: Spring 2023
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import export_text

import myFL

dataPath = "data/westermo"
dataBottom = pd.read_csv(dataPath + "/output_bottom.csv")
dataLeft = pd.read_csv(dataPath + "/output_left.csv")
dataRight = pd.read_csv(dataPath + "/output_right.csv")

### DATA PREPROCESSING

# Assigning column names
column_names = ["sAddress",
           "rAddress",
           "sMACs",
           "rMACs",
           "sIPs",
           "rIPs",
           "protocol",
           "startDate",
           "endDate",
           "start",
           "end",
           "startOffset",
           "endOffset",
           "duration",
           "sPackets",
           "rPackets",
           "sBytesSum",
           "rBytesSum",
           "sBytesMax",
           "rBytesMax",
           "sBytesMin",
           "rBytesMin",
           "sBytesAvg",
           "rBytesAvg",
           "sLoad",
           "rLoad",
           "sPayloadSum",
           "rPayloadSum",
           "sPayloadMax",
           "rPayloadMax",
           "sPayloadMin",
           "rPayloadMin",
           "sPayloadAvg",
           "rPayloadAvg",
           "sInterPacketAvg",
           "rInterPacketAvg",
           "sttl",
           "rttl",
           "sAckRate",
           "rAckRate",
           "sUrgRate",
           "rUrgRate",
           "sFinRate",
           "rFinRate",
           "sPshRate",
           "rPshRate",
           "sSynRate",
           "rSynRate",
           "sRstRate",
           "rRstRate",
           "sWinTCP",
           "rWinTCP",
           "sFragmentRate",
           "rFragmentRate",
           "sAckDelayMax",
           "rAckDelayMax",
           "sAckDelayMin",
           "rAckDelayMin",
           "sAckDelayAvg",
           "rAckDelayAvg",
           "IT_B_Label",
           "IT_M_Label",
           "NST_B_Label",
           "NST_M_Label"
           ]
dataBottom.columns = column_names
dataLeft.columns = column_names
dataRight.columns = column_names

# Removing unusable column
remove_columns = ["startDate", "endDate", "start", "end", "startOffset", "endOffset",
                  "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs",
                  "IT_B_Label", "IT_M_Label", "NST_B_Label"] # Other labels not used
dataBottom = dataBottom.drop(columns = remove_columns)
dataLeft = dataLeft.drop(columns = remove_columns)
dataRight = dataRight.drop(columns = remove_columns)

# Filling empty cells
dataBottom = dataBottom.replace([np.inf, -np.inf], -1)
dataLeft = dataLeft.replace([np.inf, -np.inf], -1)
dataRight = dataRight.replace([np.inf, -np.inf], -1)
dataBottom = dataBottom.replace(np.nan, -1)
dataLeft = dataLeft.replace(np.nan, -1)
dataRight = dataRight.replace(np.nan, -1)

# GOOD-SSH -> Normal
indr = dataRight["NST_M_Label"] == 'GOOD-SSH'
dataRight.loc[indr, "NST_M_Label"] = 'Normal'
indl = dataLeft["NST_M_Label"] == 'GOOD-SSH'
dataLeft.loc[indl, "NST_M_Label"] = 'Normal'
indb = dataBottom["NST_M_Label"] == 'GOOD-SSH'
dataBottom.loc[indb, "NST_M_Label"] = 'Normal'

# Extracting label column
boty = dataBottom["NST_M_Label"]
botx = dataBottom.drop(columns=["NST_M_Label"], axis = 1)
lefty = dataLeft["NST_M_Label"]
leftx = dataLeft.drop(columns=["NST_M_Label"], axis = 1)
righty = dataRight["NST_M_Label"]
rightx = dataRight.drop(columns=["NST_M_Label"], axis = 1)

# Attack detection without classification
boty = np.where(boty == "Normal", 0, 1)
lefty = np.where(lefty == "Normal", 0, 1)
righty = np.where(righty == "Normal", 0, 1)

# Encoding protocol column
label_encoder = LabelEncoder()
protocols = pd.concat([leftx['protocol'], botx['protocol'], rightx['protocol']])
label_encoder.fit(protocols)

botx['protocol'] = label_encoder.transform(botx['protocol'])
botxn = botx.values

leftx['protocol'] = label_encoder.transform(leftx['protocol'])
leftxn = leftx.values

rightx['protocol'] = label_encoder.transform(rightx['protocol'])
rightxn = rightx.values

# Splitting the dataset
# Train =  70%, Validate = 10%, Test = 20%
x_train_bottom, x_rest_bottom, y_train_bottom, y_rest_bottom = train_test_split(botxn, boty, test_size = 0.3, stratify=boty, random_state=42)
x_validate_bottom, x_test_bottom, y_validate_bottom, y_test_bottom = train_test_split(x_rest_bottom, y_rest_bottom, test_size = 0.67, stratify=y_rest_bottom, random_state=42)

x_train_left, x_rest_left, y_train_left, y_rest_left = train_test_split(leftxn, lefty, test_size = 0.3, stratify=lefty, random_state=42)
x_validate_left, x_test_left, y_validate_left, y_test_left = train_test_split(x_rest_left, y_rest_left, test_size = 0.67, stratify=y_rest_left, random_state=42)

x_train_right, x_rest_right, y_train_right, y_rest_right = train_test_split(rightxn, righty, test_size = 0.3, stratify=righty, random_state=42)
x_validate_right, x_test_right, y_validate_right, y_test_right = train_test_split(x_rest_right, y_rest_right, test_size = 0.67, stratify=y_rest_right, random_state=42)

# Normalizing in order to prevent "large value"-features to have greater importance.
min_max_scaler = MinMaxScaler().fit(x_train_bottom)
min_max_scaler = MinMaxScaler().fit(x_train_left)
min_max_scaler = MinMaxScaler().fit(x_train_right)

x_train_bottom = min_max_scaler.transform(x_train_bottom)
x_validate_bottom = min_max_scaler.transform(x_validate_bottom)
x_test_bottom = min_max_scaler.transform(x_test_bottom)
x_train_left = min_max_scaler.transform(x_train_left)
x_validate_left = min_max_scaler.transform(x_validate_left)
x_test_left = min_max_scaler.transform(x_test_left)
x_train_right = min_max_scaler.transform(x_train_right)
x_validate_right = min_max_scaler.transform(x_validate_right)
x_test_right = min_max_scaler.transform(x_test_right)

### END OF DATA PREPROCESSING

# Final test-set used for all experiments
x_test = np.concatenate([x_test_bottom, x_test_left, x_test_right])
y_test = np.concatenate([y_test_bottom, y_test_left, y_test_right])

# Concatinated training and validation-set used for Non-FL
x_train = np.concatenate([x_train_bottom, x_train_left, x_train_right])
y_train = np.concatenate([y_train_bottom, y_train_left, y_train_right])
x_validate = np.concatenate([x_validate_bottom, x_validate_left, x_validate_right])
y_validate = np.concatenate([y_validate_bottom, y_validate_left, y_validate_right])

clients = {
    "left": {"train": {"x": x_train_left, "y": y_train_left}, "validate": {"x": x_validate_left, "y": y_validate_left}, "test": {"x": x_test_left, "y": y_test_left}},
    "right": {"train": {"x": x_train_right, "y": y_train_right}, "validate": {"x": x_validate_right, "y": y_validate_right}, "test": {"x": x_test_right, "y": y_test_right}},
    "bottom": {"train": {"x": x_train_bottom, "y": y_train_bottom}, "validate": {"x": x_validate_bottom, "y": y_validate_bottom}, "test": {"x": x_test_bottom, "y": y_test_bottom}}
}
# SET ANALYSIS
#with open("setAnalysisOutput.txt", "w") as f:
#    for client, datasets in clients.items():
#        print("\nClient:", client, file=f)
#        for dataset, data in datasets.items():
#            y = data["y"]
#            unique_classes, class_counts = np.unique(y, return_counts=True)
#            for cls, count in zip(unique_classes, class_counts):
#                print(dataset, "Class:", cls, "Count:", count, file=f)

# 1: Traditional Random Forests
# 2: Differentially private Random Forests
# 3: Traditional Random Forests with Federated Learning
# 4: Differentially Private Random Forests with Federated Learning
# 5: Client-wise Traditional Random Forests
# 6: Client-wise Differentially Private Random Forests
expToRun = [True, True, True, True, True, True]

# 1: Traditinoal Random Forests
if(expToRun[0]):
    print("Running exp 1!")
    with open("expOutput/1output.txt", "w") as f:
        print("EXPERIMENT 1: Traditional Random Forests. All data is merged and a RF is trained and tested on the complete data.", file=f)
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11, None]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
        scores = []

        for param in params:
            RF = RandomForestClassifier(**param)
            RF.fit(x_train, y_train)

            pred = RF.predict(x_validate)
            cm = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
            auc = roc_auc_score(y_validate, pred)
            scores.append({'AUC': auc, 'Params': param, "Model" : RF})

        sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
        bestRF = sortedScores[0]["Model"]
        # Final prediction
        finalPred = bestRF.predict(x_test)
        finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
        finalAUC = roc_auc_score(y_test, finalPred)

        # To file
        print('--------------------------------------------------------------------------------------', file=f)
        print('PARAMS: ' + str(bestRF.get_params()), file=f)
        print('complete_test_AUC: ' + str(finalAUC), file=f)
        print('COMPLETE_TEST_CM: ', file=f)
        print(finalCM, file=f)
        print('--------------------------------------------------------------------------------------', file=f)



# 2: Differential Private Random Forests
warnings.filterwarnings("ignore", category=PrivacyLeakWarning)
warnings.filterwarnings("ignore", category=DiffprivlibCompatibilityWarning)
if(expToRun[1]):
    print("Running exp 2!")
    with open("expOutput/2output.txt", "w") as f:
        print("EXPERIMENT 2: Differential Private Random Forests. All data is merged and a DP_RF is trained and tested on the complete data.", file=f)
        eps = [0.01, 0.1, 0.5, 1.0, 5.0, 10, 50, 100]
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]

        bestModels = []

        for e in eps:
            print("Using eps = " + str(e))
            scores = []
            for param in params:
                DP_RF = DP_RandomForestClassifier(epsilon=e, **param)
                DP_RF.fit(x_train, y_train)

                pred = DP_RF.predict(x_validate)
                cm = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
                auc = roc_auc_score(y_validate, pred)
                scores.append({'AUC': auc, 'Params': param, "Model" : DP_RF})

            sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
            bestModels.append(sortedScores[0]["Model"])
        # Final predictions
        for RF in bestModels:
            finalPred = RF.predict(x_test)
            finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
            finalAUC = roc_auc_score(y_test, finalPred)

            # To file
            print('--------------------------------------------------------------------------------------', file=f)
            print('EPS: ' + str(RF.epsilon), file=f)
            print('PARAMS: ' + str(RF.get_params()), file=f)
            print('complete_test_AUC: ' + str(finalAUC), file=f)
            print('COMPLETE_TEST_CM: ', file=f)
            print(finalCM, file=f)
            print('--------------------------------------------------------------------------------------', file=f)

# 3: Federated traditional Random Forests
if(expToRun[2]):
    print("Running exp 3!")
    with open("expOutput/3output.txt", "w") as f:
        print("EXPERIMENT 3: Federated traditional Random Forests. Each client trains on own data is then merged and asked to predict on complete test-set", file=f)
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11, None]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
        bestRF = [] # The best RF from each client

        for client in clients:
            scores = []
            for param in params:
                RF = RandomForestClassifier(**param)
                RF.fit(clients[client]["train"]["x"], clients[client]["train"]["y"])

                pred = RF.predict(clients[client]["validate"]["x"])
                cm = pd.crosstab(pd.Series(clients[client]["validate"]["y"], name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
                auc = roc_auc_score(clients[client]["validate"]["y"], pred)
                scores.append({'AUC': auc, 'Params': param, "Model" : RF})

            sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
            bestRF.append(sortedScores[0]["Model"])
            print('START---------------------------------------------------------------------------------', file=f)
            print('client: '+client, file=f)
            print('PARAMS: '+str(sortedScores[0]["Model"].get_params()), file=f)
            print('--------------------------------------------------------------------------------------', file=f)
        # Merging
        FLRF = myFL.FL_Forest()
        mergedForest = FLRF.mergeALL(bestRF)
        # Final prediction
        finalPred = mergedForest.predict(x_test)
        finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
        finalAUC = roc_auc_score(y_test, finalPred)

        # To file
        print('--------------------------------------------------------------------------------------', file=f)
        print('PARAMS: ' + str(mergedForest.get_params()), file=f)
        print('complete_test_AUC: ' + str(finalAUC), file=f)
        print('COMPLETE_TEST_CM: ', file=f)
        print(finalCM, file=f)
        print('END-----------------------------------------------------------------------------------', file=f)

# 4: Federated Differential Private Random Forests
if(expToRun[3]):
    print("Running exp 4!")
    with open("expOutput/4output.txt", "w") as f:
        print("EXPERIMENT 4: Federated Differential Private Random Forests. Each client trains on own data is then merged and asked to predict on complete test-set", file=f)
        eps = [0.01, 0.1, 0.5, 1.0, 5.0, 10, 50, 100]
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
        bestModels = []
        for e in eps:
            print("Using eps = " + str(e))
            bestRF = []  # The best RF from each client
            for client in clients:
                scores = []
                for param in params:
                    RF = DP_RandomForestClassifier(epsilon = e, **param)
                    RF.fit(clients[client]["train"]["x"], clients[client]["train"]["y"])

                    pred = RF.predict(clients[client]["validate"]["x"])
                    cm = pd.crosstab(pd.Series(clients[client]["validate"]["y"], name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
                    auc = roc_auc_score(clients[client]["validate"]["y"], pred)
                    scores.append({'AUC': auc, 'Params': param, "Model": RF})

                sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
                bestRF.append(sortedScores[0]["Model"])
                print('START---------------------------------------------------------------------------------', file=f)
                print('client: ' + client, file=f)
                print('PARAMS: ' + str(sortedScores[0]["Model"].get_params()), file=f)
                print('--------------------------------------------------------------------------------------', file=f)
            # Merging
            FLRF = myFL.FL_DP_Forest()
            mergedForest = FLRF.mergeALL(bestRF)
            bestModels.append(mergedForest)
        # Final predictions
        for RF in bestModels:
            finalPred = RF.predict(x_test)
            finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
            finalAUC = roc_auc_score(y_test, finalPred)

            # To file
            print('--------------------------------------------------------------------------------------', file=f)
            print('EPS: ' + str(RF.epsilon), file=f)
            print('PARAMS: ' + str(RF.get_params()), file=f)
            print('complete_test_AUC: ' + str(finalAUC), file=f)
            print('COMPLETE_TEST_CM: ', file=f)
            print(finalCM, file=f)
            print('END-----------------------------------------------------------------------------------', file=f)

# 5: Traditinoal Random Forests client-wise
if(expToRun[4]):
    print("Running exp 5!")
    with open("expOutput/5output.txt", "w") as f:
        print("EXPERIMENT 5: Client-wise training and evaluation on client- and complete test set", file=f)
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11, None]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]

        for client in clients:
            scores = []
            for param in params:
                RF = RandomForestClassifier(**param)
                RF.fit(clients[client]["train"]["x"], clients[client]["train"]["y"])
                pred = RF.predict(clients[client]["validate"]["x"])
                cm = pd.crosstab(pd.Series(clients[client]["validate"]["y"], name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
                auc = roc_auc_score(clients[client]["validate"]["y"], pred)
                scores.append({'AUC': auc, 'Params': param, "Model" : RF})
            sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
            bestRF = sortedScores[0]["Model"] # Best model of client
            # Prediction on client_test_set
            client_testPred = bestRF.predict(clients[client]["test"]["x"])
            complete_testPred = bestRF.predict(x_test)
            client_test_AUC = roc_auc_score(clients[client]["test"]["y"], client_testPred)
            complete_test_AUC = roc_auc_score(y_test, complete_testPred)
            client_test_CM = pd.crosstab(pd.Series(clients[client]["test"]["y"], name="Actual"), pd.Series(client_testPred, name="Predicted"),normalize="all")
            complete_test_CM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(complete_testPred, name="Predicted"),normalize="all")

            # To file
            print('--------------------------------------------------------------------------------------', file=f)
            print('CLIENT: ' + str(client), file=f)
            print('PARAMS: '+ str(bestRF.get_params()), file=f)
            print('client_test_AUC: ' + str(client_test_AUC), file=f)
            print('complete_test_AUC: ' + str(complete_test_AUC), file=f)
            print('CLIENT_TEST_CM: ', file=f)
            print(client_test_CM, file=f)
            print('\nCOMPLETE_TEST_CM: ', file=f)
            print(complete_test_CM, file=f)
            print('--------------------------------------------------------------------------------------', file=f)
# 6: Differential Private Random Forests client-wise
if(expToRun[5]):
    print("Running exp 6!")
    with open("expOutput/6output.txt", "w") as f:
        print("EXPERIMENT 6: Client-wise DP training and evaluation on client- and complete test set", file=f)
        eps = [0.01, 0.1, 0.5, 1.0, 5.0, 10, 50, 100]
        n_trees = [10, 50, 100, 200, 400]
        m_depths = [3, 5, 7, 11]
        m_samples = [0.7, 0.8, 0.9, None]

        params_list = list(itertools.product(n_trees, m_depths, m_samples))
        params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
        for e in eps:
            print("Using eps = "+str(e))
            for client in clients:
                scores = []
                for param in params:
                    RF = DP_RandomForestClassifier(epsilon = e, **param)
                    RF.fit(clients[client]["train"]["x"], clients[client]["train"]["y"])

                    pred = RF.predict(clients[client]["validate"]["x"])
                    cm = pd.crosstab(pd.Series(clients[client]["validate"]["y"], name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
                    auc = roc_auc_score(clients[client]["validate"]["y"], pred)
                    scores.append({'AUC': auc, 'Params': param, "Model" : RF})

                sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
                bestRF = sortedScores[0]["Model"] # Best model of client
                # Prediction on client_test_set
                client_testPred = bestRF.predict(clients[client]["test"]["x"])
                complete_testPred = bestRF.predict(x_test)
                client_test_AUC = roc_auc_score(clients[client]["test"]["y"], client_testPred)
                complete_test_AUC = roc_auc_score(y_test, complete_testPred)
                client_test_CM = pd.crosstab(pd.Series(clients[client]["test"]["y"], name="Actual"), pd.Series(client_testPred, name="Predicted"),normalize="all")
                complete_test_CM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(complete_testPred, name="Predicted"),normalize="all")

                # To file
                print('--------------------------------------------------------------------------------------', file=f)
                print('CLIENT: ' + str(client), file=f)
                print('EPS: ' + str(RF.epsilon), file=f)
                print('PARAMS: '+ str(bestRF.get_params()), file=f)
                print('client_test_AUC: ' + str(client_test_AUC), file=f)
                print('complete_test_AUC: ' + str(complete_test_AUC), file=f)
                print('CLIENT_TEST_CM: ', file=f)
                print(client_test_CM, file=f)
                print('\nCOMPLETE_TEST_CM: ', file=f)
                print(complete_test_CM, file=f)
                print('--------------------------------------------------------------------------------------', file=f)