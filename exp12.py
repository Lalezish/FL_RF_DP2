# Author: Alexander Frid
# Date: Spring 2023
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
x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size = 0.3)
x_validate, x_test, y_validate, y_test = train_test_split(x_rest, y_rest, test_size = 0.67)

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
# 1: Traditional Random Forests with 10, 50, 100, 200, 400 trees
# 2: Differential Private Random Forests with epsilon 0.01, 0.1, 0.5, 1.0, 5.0

# 1: Traditinoal Random Forests
with open("expOutput/1output.txt", "w") as f:
    n_trees = [10, 50, 100, 200, 400]
    for n in n_trees:
        RF = RandomForestClassifier(n_estimators=n)
        RF.fit(x_train, y_train)
        pred = RF.predict(x_test)
        cm = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
        auc = roc_auc_score(y_test, pred)
        print('AUC for the test-set with '+str(n)+" trees")
        print(auc)
        print('Confusion matrix for the test set RF with '+str(n)+" trees")
        print(cm)
        print('Confusion matrix for the test set RF with '+str(n)+" trees", file=f)
        print(auc, file=f)
        print(cm, file=f)

# 2: Differential Private Random Forests
with open("expOutput/2output.txt", "w") as f:
    eps = [0.01, 0.1, 0.5, 1.0, 5.0, float('inf')]
    for e in eps:
        DP_RF = DP_RandomForestClassifier(epsilon=e, n_estimators=400)
        DP_RF.fit(x_train, y_train)
        pred = DP_RF.predict(x_test)
        cm = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(pred, name="Predicted"), normalize="all")
        auc = roc_auc_score(y_test, pred)
        print('AUC for the test-set with epsilon: ' + str(e))
        print(auc)
        print('Confusion matrix for the test set DP_RF with epsilon: ' + str(e))
        print(cm)
        print('Confusion matrix for the test set DP_RF with epsilon: ' + str(e), file=f)
        print(auc, file=f)
        print(cm, file=f)