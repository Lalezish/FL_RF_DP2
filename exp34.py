# Author: Alexander Frid
# Date: Spring 2023
import itertools
import time

import numpy as np
import pandas as pd
from diffprivlib.utils import PrivacyLeakWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
                  "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "protocol",
                  "IT_B_Label", "IT_M_Label", "NST_B_Label"] # Other labels not used for now
dataBottom = dataBottom.drop(columns = remove_columns)
dataLeft = dataLeft.drop(columns = remove_columns)
dataRight = dataRight.drop(columns = remove_columns)

# Removing entires which contain inf, -inf or none values.
dataBottom = dataBottom.replace([np.inf, -np.inf], np.nan)
dataLeft = dataLeft.replace([np.inf, -np.inf], np.nan)
dataRight = dataRight.replace([np.inf, -np.inf], np.nan)
dataBottom = dataBottom.dropna()
dataLeft = dataLeft.dropna()
dataRight = dataRight.dropna()


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

# Splitting the dataset
# Train =  70%, Validate = 10%, Test = 20%
x_train_bottom, x_rest_bottom, y_train_bottom, y_rest_bottom = train_test_split(botx, boty, test_size = 0.3, stratify=boty, random_state=42)
x_validate_bottom, x_test_bottom, y_validate_bottom, y_test_bottom = train_test_split(x_rest_bottom, y_rest_bottom, test_size = 0.67, stratify=y_rest_bottom, random_state=42)

x_train_left, x_rest_left, y_train_left, y_rest_left = train_test_split(leftx, lefty, test_size = 0.3, stratify=lefty, random_state=42)
x_validate_left, x_test_left, y_validate_left, y_test_left = train_test_split(x_rest_left, y_rest_left, test_size = 0.67, stratify=y_rest_left, random_state=42)

x_train_right, x_rest_right, y_train_right, y_rest_right = train_test_split(rightx, righty, test_size = 0.3, stratify=righty, random_state=42)
x_validate_right, x_test_right, y_validate_right, y_test_right = train_test_split(x_rest_right, y_rest_right, test_size = 0.67, stratify=y_rest_right, random_state=42)

# Dropping redundant columns
#col_names = np.array(list(x_train))
#col_to_drop = []
#for i in col_names:
#    size = x_train.groupby([i]).size()
#    if(len(size.unique()) == 1):
#        col_to_drop.append(i)

#x_train = x_train.drop(col_to_drop, axis = 1)
#x_validate = x_validate.drop(col_to_drop, axis = 1)
#x_test = x_test.drop(col_to_drop, axis = 1)

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
### Experiments
x_test = np.concatenate([x_test_bottom, x_test_left, x_test_right])
y_test = np.concatenate([y_test_bottom, y_test_left, y_test_right])
x_validate = np.concatenate([x_validate_bottom, x_validate_left, x_validate_right])
y_validate = np.concatenate([y_validate_bottom, y_validate_left, y_validate_right])
# 3: Federated traditional Random Forests
with open("expOutput/3output.txt", "w") as f:
    n_trees = [10, 50, 100, 200, 400]
    m_depths = [3, 5, 7, 11, None]
    m_samples = [0.7, 0.8, 0.9, None]

    params_list = list(itertools.product(n_trees, m_depths, m_samples))
    params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s} for n_t, m_d, m_s in params_list]
    scores = []

    for param in params:
        RF_bot = RandomForestClassifier(**param)
        RF_left = RandomForestClassifier(**param)
        RF_right = RandomForestClassifier(**param)
        RF_bot.fit(x_train_bottom, y_train_bottom)
        RF_left.fit(x_train_left, y_train_left)
        RF_right.fit(x_train_right, y_train_right)

        Forests = [RF_bot, RF_left, RF_right]
        FLRF = myFL.FL_Forest()
        RF_merged = FLRF.mergeALL(Forests)

        pred = RF_merged.predict(x_test)
        cm = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
        auc = roc_auc_score(y_test, pred)
        scores.append({'AUC': auc, 'Params': param})

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
    # To console
    print('Best Params: \n')
    for s in sortedScores:
        print(s)
    # To file
    print('Best Params: \n', file=f)
    for s in sortedScores:
        print(s)

# 4: Federated Differential Private Random Forests
with open("expOutput/4output.txt", "w") as f:
    eps = [0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
    n_trees = [10, 50, 100, 200, 400]
    m_depths = [3, 5, 7, 11]
    m_samples = [0.7, 0.8, 0.9, None]

    for param in params:
        DP_RF_bot = DP_RandomForestClassifier(**param)
        DP_RF_left = DP_RandomForestClassifier(**param)
        DP_RF_right = DP_RandomForestClassifier(**param)
        DP_RF_bot.fit(x_train_bottom, y_train_bottom)
        DP_RF_left.fit(x_train_left, y_train_left)
        DP_RF_right.fit(x_train_right, y_train_right)

        DP_Forests = [DP_RF_bot, DP_RF_left, DP_RF_right]
        FLDPRF = myFL.FL_DP_Forest()
        DP_RF_merged = FLDPRF.mergeALL(DP_Forests)

        pred = DP_RF_bot.predict(x_test)
        cm = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
        auc = roc_auc_score(y_test, pred)
        scores.append({'AUC': auc, 'Params': param})

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
    # To console
    print('Best Params: \n')
    for s in sortedScores:
        print(s)
    # To file
    print('Best Params: \n', file=f)
    for s in sortedScores:
        print(s)