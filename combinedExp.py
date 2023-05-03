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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
feature = onehot_encoder.fit_transform(botx['protocol'].values.reshape(-1, 1))
botxn = botx.drop(columns=['protocol']).values
botxn = np.concatenate((feature, botxn), axis=1)

feature = onehot_encoder.transform(leftx['protocol'].values.reshape(-1, 1))
leftxn = leftx.drop(columns=['protocol']).values
leftxn = np.concatenate((feature, leftxn), axis=1)

feature = onehot_encoder.transform(rightx['protocol'].values.reshape(-1, 1))
rightxn = rightx.drop(columns=['protocol']).values
rightxn = np.concatenate((feature, rightxn), axis=1)

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

### Non-FL experiments
# 1: Traditional Random Forests
# 2: Differential Private Random Forests

# 1: Traditinoal Random Forests
with open("expOutput/1output.txt", "w") as f:
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
        # To console
        print('PARAMS: ' + str(param))
        print('AUC: ' + str(auc))
        print(cm)
        print('--------------------------------------------------------------------------------------')

        # To file
        print('PARAMS: ' + str(param), file=f)
        print('AUC: ' + str(auc), file=f)
        print(cm, file=f)
        print('--------------------------------------------------------------------------------------', file=f)

    sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
    bestRF = sortedScores[0]["Model"]
    # Final prediction
    finalPred = bestRF.predict(x_test)
    finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
    finalAUC = roc_auc_score(y_test, finalPred)
    # To console
    print('-------------------------------------FINAL MODEL--------------------------------------')

    print('AUC: ' + str(finalAUC))
    print(finalCM)
    print('--------------------------------------------------------------------------------------')

    # To file
    print('-------------------------------------FINAL MODEL--------------------------------------', file=f)
    print('AUC: ' + str(finalAUC), file=f)
    print(finalCM, file=f)
    print('--------------------------------------------------------------------------------------', file=f)



# 2: Differential Private Random Forests
warnings.filterwarnings("ignore", category=PrivacyLeakWarning)
warnings.filterwarnings("ignore", category=DiffprivlibCompatibilityWarning)

with open("expOutput/2output.txt", "w") as f:
    eps = [0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
    n_trees = [10, 50, 100, 200, 400]
    m_depths = [3, 5, 7, 11]
    m_samples = [0.7, 0.8, 0.9, None]

    params_list = list(itertools.product(n_trees, m_depths, m_samples))
    params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]

    bestModels = []

    for e in eps:
        scores = []
        for param in params:
            DP_RF = DP_RandomForestClassifier(epsilon=e, **param)
            DP_RF.fit(x_train, y_train)

            pred = DP_RF.predict(x_validate)
            cm = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
            auc = roc_auc_score(y_validate, pred)
            scores.append({'AUC': auc, 'Params': param, "Model" : RF})

            # To console
            print('PARAMS: ' + str(param))
            print('AUC: ' + str(auc))
            print(cm)
            print('--------------------------------------------------------------------------------------')

            # To file
            print('PARAMS: ' + str(param), file=f)
            print('AUC: ' + str(auc), file=f)
            print(cm, file=f)
            print('--------------------------------------------------------------------------------------', file=f)

        sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
        bestModels.append(sortedScores[0]["Model"])
    # Final predictions
    for RF in bestModels:
        finalPred = RF.predict(x_test)
        finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
        finalAUC = roc_auc_score(y_test, finalPred)
        # To console
        print('-------------------------------------FINAL MODEL--------------------------------------')
        print('Params: ' + str(RF.get_params()))
        print('AUC: ' + str(finalAUC))
        print(finalCM)
        print('--------------------------------------------------------------------------------------')

        # To file
        print('-------------------------------------FINAL MODEL--------------------------------------', file=f)

        print('AUC: ' + str(finalAUC), file=f)
        print(finalCM, file=f)
        print('--------------------------------------------------------------------------------------', file=f)

### FL experiments

# 3: Federated traditional Random Forests
# 4: Federated Differential Private Random Forests

clients = {
    "left": {"train": {"x": x_train_left, "y": y_train_left}, "validate": {"x": x_validate_left, "y": y_validate_left}},
    "right": {"train": {"x": x_train_right, "y": y_train_right}, "validate": {"x": x_validate_right, "y": y_validate_right}},
    "bottom": {"train": {"x": x_train_bottom, "y": y_train_bottom}, "validate": {"x": x_validate_bottom, "y": y_validate_bottom}},
}

# 3: Federated traditional Random Forests
with open("expOutput/3output.txt", "w") as f:
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

            # To console
            print('PARAMS: ' + str(param))
            print('AUC: ' + str(auc))
            print(cm)
            print('--------------------------------------------------------------------------------------')

            # To file
            print('PARAMS: ' + str(param), file=f)
            print('AUC: ' + str(auc), file=f)
            print(cm, file=f)
            print('--------------------------------------------------------------------------------------', file=f)

        sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
        bestRF.append(sortedScores[0]["Model"])
    # Merging
    FLRF = myFL.FL_Forest()
    mergedForest = FLRF.mergeALL(bestRF)
    # Final prediction
    finalPred = mergedForest.predict(x_test)
    finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
    finalAUC = roc_auc_score(y_test, finalPred)
    # To console
    print('-------------------------------------FINAL MODEL--------------------------------------')

    print('AUC: ' + str(finalAUC))
    print(finalCM)
    print('--------------------------------------------------------------------------------------')

    # To file
    print('-------------------------------------FINAL MODEL--------------------------------------', file=f)
    print('AUC: ' + str(finalAUC), file=f)
    print(finalCM, file=f)
    print('--------------------------------------------------------------------------------------', file=f)

# 4: Federated Differential Private Random Forests
with open("expOutput/4output.txt", "w") as f:
    eps = [0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
    n_trees = [10, 50, 100, 200, 400]
    m_depths = [3, 5, 7, 11]
    m_samples = [0.7, 0.8, 0.9, None]

    params_list = list(itertools.product(n_trees, m_depths, m_samples))
    params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
    bestModels = []
    for e in eps:
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

                # To console
                print('PARAMS: ' + str(param))
                print('AUC: ' + str(auc))
                print(cm)
                print('--------------------------------------------------------------------------------------')

                # To file
                print('PARAMS: ' + str(param), file=f)
                print('AUC: ' + str(auc), file=f)
                print(cm, file=f)
                print('--------------------------------------------------------------------------------------', file=f)

            sortedScores = sorted(scores, key=lambda x: x['AUC'], reverse=True)
            bestRF.append(sortedScores[0]["Model"])
        # Merging
        FLRF = myFL.FL_DP_Forest()
        mergedForest = FLRF.mergeALL(bestRF)
        bestModels.append(mergedForest)
    # Final predictions
    for RF in bestModels:
        finalPred = RF.predict(x_test)
        finalCM = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(finalPred, name="Predicted"), normalize="all")
        finalAUC = roc_auc_score(y_test, finalPred)
        # To console
        print('-------------------------------------FINAL MODEL--------------------------------------')
        print('EPS: ' + str(RF.epsilon))
        print('AUC: ' + str(finalAUC))
        print(finalCM)
        print('--------------------------------------------------------------------------------------')

        # To file
        print('-------------------------------------FINAL MODEL--------------------------------------', file=f)

        print('AUC: ' + str(finalAUC), file=f)
        print(finalCM, file=f)
        print('--------------------------------------------------------------------------------------', file=f)