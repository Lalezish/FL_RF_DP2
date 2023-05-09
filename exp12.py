# Author: Alexander Frid
# Date: Spring 2023
import itertools
import warnings

import numpy as np
import pandas as pd
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataPath = "data/westermo"
dataBottom = pd.read_csv(dataPath + "/output_bottom.csv")
dataLeft = pd.read_csv(dataPath + "/output_left.csv")
dataRight = pd.read_csv(dataPath + "/output_right.csv")
data = pd.concat([dataBottom, dataLeft, dataRight])

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
data.columns = column_names

# Removing unusable column
remove_columns = ["startDate", "endDate", "start", "end", "startOffset", "endOffset",
                  "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "protocol",
                  "IT_B_Label", "IT_M_Label", "NST_B_Label"] # Other labels not used for now
data = data.drop(columns = remove_columns)

# Removing entires which contain inf, -inf or none values.
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()


# Extracting label column
y = data["NST_M_Label"]
x = data.drop(columns=["NST_M_Label"], axis = 1)

# Attack detection without classification
y = np.where(y == "Normal", 0, 1)

# Splitting the dataset
# Train =  70%, Validate = 10%, Test = 20%
x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size = 0.3, stratify=y, random_state=42)
x_validate, x_test, y_validate, y_test = train_test_split(x_rest, y_rest, test_size = 0.67, stratify=y_rest, random_state=42)

# Dropping redundant columns
col_names = np.array(list(x_train))
col_to_drop = []
for i in col_names:
    size = x_train.groupby([i]).size()
    if(len(size.unique()) == 1):
        col_to_drop.append(i)

x_train = x_train.drop(col_to_drop, axis = 1)
x_validate = x_validate.drop(col_to_drop, axis = 1)
x_test = x_test.drop(col_to_drop, axis = 1)

# Normalizing in order to prevent "large value"-features to have greater importance.
min_max_scaler = MinMaxScaler().fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_validate = min_max_scaler.transform(x_validate)
x_test = min_max_scaler.transform(x_test)

### END OF DATA PREPROCESSING

### Experiments
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
        scores.append({'AUC': auc, 'Params': param})
        
        # To console
        print('PARAMS: '+str(param))
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

# 2: Differential Private Random Forests
warnings.filterwarnings("ignore", category=PrivacyLeakWarning)
warnings.filterwarnings("ignore", category=DiffprivlibCompatibilityWarning)

with open("expOutput/2output.txt", "w") as f:
    eps = [0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
    n_trees = [10, 50, 100, 200, 400]
    m_depths = [3, 5, 7, 11]
    m_samples = [0.7, 0.8, 0.9, None]

    params_list = list(itertools.product(n_trees, m_depths, m_samples, eps))
    params = [{'n_estimators': n_t, 'max_depth': m_d, 'max_samples': m_s, 'epsilon': e} for n_t, m_d, m_s, e in params_list]
    scores = []

    for param in params:
        DP_RF = DP_RandomForestClassifier(**param)
        DP_RF.fit(x_train, y_train)

        pred = DP_RF.predict(x_validate)
        cm = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
        auc = roc_auc_score(y_validate, pred)
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