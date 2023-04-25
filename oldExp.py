# Author: Alexander Frid
# Date: Spring 2023
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

# Classification of attack:
#le = LabelEncoder()
#le.fit(y)
#y = le.transform(y)

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

### MODEL CREATION
# using: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

depth = None
m_features = 1.0
jobs = -1
bs = True

classifier = RandomForestClassifier(n_estimators=400, max_depth=depth, max_features=m_features, n_jobs=jobs, bootstrap=bs)
classifier.fit(x_train, y_train)

# Own experiments
ownRF1 = RandomForestClassifier(max_depth=depth, max_features=m_features, n_jobs=jobs, bootstrap=bs)
ownRF2 = RandomForestClassifier(max_depth=depth, max_features=m_features, n_jobs=jobs, bootstrap=bs)
ownRF3 = RandomForestClassifier(max_depth=depth, max_features=m_features, n_jobs=jobs, bootstrap=bs)
ownRF4 = RandomForestClassifier(max_depth=depth, max_features=m_features, n_jobs=jobs, bootstrap=bs)
x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size = 0.5)
x_train11, x_train12, y_train11, y_train12 = train_test_split(x_train1, y_train1, test_size = 0.5)
x_train21, x_train22, y_train21, y_train22 = train_test_split(x_train2, y_train2, test_size = 0.5)
ownRF1.fit(x_train11, y_train11)
ownRF2.fit(x_train12, y_train12)
ownRF3.fit(x_train21, y_train21)
ownRF4.fit(x_train22, y_train22)

# Predict on the validation set using myRF1
ownPredY1 = ownRF1.predict(x_validate)
ownConfusionMatrix1 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(ownPredY1, name="Predicted"))
print('Confusion matrix for the validation set (myRF) 1')
print(ownConfusionMatrix1)

# Predict on the validation set using myRF2
ownPredY2 = ownRF2.predict(x_validate)
ownConfusionMatrix2 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(ownPredY2, name="Predicted"))
print('Confusion matrix for the validation set (myRF) 2')
print(ownConfusionMatrix2)

# Predict on the validation set using myRF3
ownPredY3 = ownRF3.predict(x_validate)
ownConfusionMatrix3 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(ownPredY3, name="Predicted"))
print('Confusion matrix for the validation set (myRF) 3')
print(ownConfusionMatrix3)

# Predict on the validation set using myRF4
ownPredY4 = ownRF4.predict(x_validate)
ownConfusionMatrix4 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(ownPredY4, name="Predicted"))
print('Confusion matrix for the validation set (myRF) 4')
print(ownConfusionMatrix4)

# Predict on the validation set using RandomForestClassifier
y_predicted_validate = classifier.predict(x_validate)
confusion_matrix_validation = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(y_predicted_validate, name="Predicted"))
print('\nConfusion matrix for the validation set (RandomForestClassifier)')
print(confusion_matrix_validation)

# Merged Forest
forestsToMerge = [ownRF1, ownRF2, ownRF3, ownRF4]
FLForest = myFL.FL_Forest()
#newForest = FLForest.merge(forestsToMerge)
newForest = FLForest.mergeAccuracy(forestsToMerge, 10, x_test, y_test)

newForestPred = newForest.predict(x_validate)
newConfusionMatrix = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(newForestPred, name="Predicted"))
print('\nConfusion matrix for the validation set (Merged Forest)')
print(newConfusionMatrix)

accuracy = classifier.score(x_train, y_train)
print("Accuracy (Classifier, Train): {:.5f}".format(accuracy))

accuracy2 = classifier.score(x_validate, y_validate)
print("Accuracy (Classifier, Validate): {:.5f}".format(accuracy2))

accuracyMerge = newForest.score(x_train, y_train)
print("Accuracy (MergedForest, Train): {:.5f}".format(accuracyMerge))

accuracyMerge2 = newForest.score(x_validate, y_validate)
print("Accuracy (MergedForest, Validate): {:.5f}".format(accuracyMerge2))


accuracy1t = ownRF1.score(x_train, y_train)
accuracy1 = ownRF1.score(x_validate, y_validate)
accuracy2t = ownRF2.score(x_train, y_train)
accuracy2 = ownRF2.score(x_validate, y_validate)
accuracy3t = ownRF3.score(x_train, y_train)
accuracy3 = ownRF3.score(x_validate, y_validate)
accuracy4t = ownRF4.score(x_train, y_train)
accuracy4 = ownRF4.score(x_validate, y_validate)
print("Accuracy (MergedForest 1, Train): {:.5f}".format(accuracy1t))
print("Accuracy (MergedForest 1, Validate): {:.5f}".format(accuracy1))
print("Accuracy (MergedForest 2, Train): {:.5f}".format(accuracy2t))
print("Accuracy (MergedForest 2, Validate): {:.5f}".format(accuracy2))
print("Accuracy (MergedForest 3, Train): {:.5f}".format(accuracy3t))
print("Accuracy (MergedForest 3, Validate): {:.5f}".format(accuracy3))
print("Accuracy (MergedForest 4, Train): {:.5f}".format(accuracy4t))
print("Accuracy (MergedForest 4, Validate): {:.5f}".format(accuracy4))

# IBM diffpriv
"""A smaller ε will yield better privacy but less accurate response. Small values of ε require to provide very similar outputs when given similar inputs,
    and therefore provide higher levels of privacy; large values of ε allow less similarity in the outputs, and therefore provide less privacy"""
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier
IBM_Forest = DP_RandomForestClassifier(n_estimators=100, epsilon=0.1)
IBM_Forest.fit(x_train, y_train)
IBM_pred = IBM_Forest.predict(x_validate)
IBM_accuracy = IBM_Forest.score(x_validate, y_validate)
IBM_cm = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(IBM_pred, name="Predicted"))
print("Accuracy (IBM DP_FOREST, Validate): {:.5f}".format(IBM_accuracy))
print(IBM_cm)

ownDPRF1 = DP_RandomForestClassifier(n_estimators=100, epsilon=0.1)
ownDPRF2 = DP_RandomForestClassifier(n_estimators=100, epsilon=0.1)
ownDPRF3 = DP_RandomForestClassifier(n_estimators=100, epsilon=0.1)
ownDPRF4 = DP_RandomForestClassifier(n_estimators=100, epsilon=0.1)

ownDPRF1.fit(x_train11, y_train11)
ownDPRF2.fit(x_train12, y_train12)
ownDPRF3.fit(x_train21, y_train21)
ownDPRF4.fit(x_train22, y_train22)

# Predict on the validation set using myRF1
DPownPredY1 = ownDPRF1.predict(x_validate)
DPownConfusionMatrix1 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(DPownPredY1, name="Predicted"))
print('Confusion matrix for the validation set (myDPRF) 1')
print(DPownConfusionMatrix1)

# Predict on the validation set using myRF2
DPownPredY2 = ownDPRF2.predict(x_validate)
DPownConfusionMatrix2 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(DPownPredY2, name="Predicted"))
print('Confusion matrix for the validation set (myDPRF) 2')
print(DPownConfusionMatrix2)

# Predict on the validation set using myRF3
DPownPredY3 = ownDPRF3.predict(x_validate)
DPownConfusionMatrix3 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(DPownPredY3, name="Predicted"))
print('Confusion matrix for the validation set (myDPRF) 3')
print(DPownConfusionMatrix3)

# Predict on the validation set using myRF4
DPownPredY4 = ownDPRF4.predict(x_validate)
DPownConfusionMatrix4 = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(DPownPredY4, name="Predicted"))
print('Confusion matrix for the validation set (myDPRF) 4')
print(DPownConfusionMatrix4)

DPforestsToMerge = [ownDPRF1, ownDPRF2, ownDPRF3, ownDPRF4]
DPFLForest = myFL.FL_DP_Forest()
DPnewForest = DPFLForest.mergeAccuracy(DPforestsToMerge, 10, x_test, y_test)

accuracyMergeDP = DPnewForest.score(x_validate, y_validate)
print("Accuracy (MergedForest, Validate): {:.5f}".format(accuracyMergeDP))

DPnewForestPred = DPnewForest.predict(x_validate)
DPnewConfusionMatrix = pd.crosstab(pd.Series(y_validate, name="Actual"), pd.Series(DPnewForestPred, name="Predicted"))
print('\nConfusion matrix for the validation set (Merged Forest DP)')
print(DPnewConfusionMatrix)



# myRF
# RandF = myRF.RF_Classifier()
# RandF.fit(x_train, y_train, 0.1)
# RandF.predict(x_validate, y_validate)
# print(RandF.statisticsRF.confusionMatrix)

# SAM DP_FR
#SAM_train = np.column_stack((y_train, x_train))
#SAM_test = np.column_stack((y_validate, x_validate))
#SAM_Forest_train = Smooth_Random_Trees.DP_Random_Forest(SAM_train, SAM_train, [0], 400, float('inf'))
#SAM_Forest = Smooth_Random_Trees.DP_Random_Forest(SAM_train, SAM_test, [0], 400, float('inf'))
#print('accuracy (train) = '+str(SAM_Forest_train._accuracy))
#print('cm (train) = '+str(SAM_Forest_train._confusion_matrix))
#print('accuracy (test) = '+str(SAM_Forest._accuracy))
#print('cm (test) = '+str(SAM_Forest._confusion_matrix))

#accuracy (train) = 0.8038491008248073
#accuracy (test) = 0.8367867389225374

# eps 0 -> ~0.55
# eps 0.1 -> 0.8622888109658909
# eps 1.0 -> 0.8629263627669748
# eps 7000.0 -> 0.8724896397832324
# eps inf -> 0.8728084156837743

# DP_RF
#dp_forest = DPForest.DPRandomForest(400, "gini", "SV", 2, epsilon=0.1)
#dp_forest.fit(x_train, y_train)
#dp_forest.predict(x_validate, y_validate)
#print(dp_forest.statisticsRF.myConfusionMatrix)

# Old RF-implementation
#print('\nConfusion matrix for the validation set (THEIR-FOREST)')
#theirRF = RandomForest.RandomForest(100, "gini", "SV", 2)
#theirRF.fit(x_train, y_train, 1)
#theirRF.predict(x_validate, y_validate)
#print(theirRF.statisticsRF.myConfusionMatrix)

# TREE-FEATURE ANALYSIS

# Tree Depth
#for i, tree in enumerate(newForest.estimators_):
#    print('Tree', i+1, 'depth:', tree.tree_.max_depth)
#
#for i, tree in enumerate(classifier.estimators_):
#    print('Tree', i+1, 'depth:', tree.tree_.max_depth)
#
#for i, (tree1, tree2) in enumerate(zip(newForest.estimators_, classifier.estimators_)):
#    depth1 = tree1.tree_.max_depth
#    depth2 = tree2.tree_.max_depth
#    depth_diff = depth1 - depth2
#    print('Tree', i+1, 'depth difference:',

# Feature Importance
#importance1 = newForest.feature_importances_
#importance2 = classifier.feature_importances_
#total_diff = 0

#for i, (imp1, imp2) in enumerate(zip(importance1, importance2)):
#    diff = abs(imp1 - imp2)
#    total_diff += diff
#print('Total difference:', total_diff)