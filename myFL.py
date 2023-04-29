"""
Federated Learning approach using multiple RandomForests in order to create a more complete ensemble of Decision Trees.
NOTE: Merging forests generated using different hyperparameters will most likely generate undesirable results.
"""

from sklearn.ensemble import RandomForestClassifier
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier
from sklearn.metrics import accuracy_score


class FL_Forest():
    def __init__(self):
        self.n_rf = 0
        self.n_estimators = 0

    # Merging ALL the Trees present in ALL of the Forests
    def mergeALL(self, input_Forests):
        params = input_Forests[0].get_params()
        self.mergedForest = RandomForestClassifier(**params)
        self.mergedForest.estimators_ = []

        for forest in input_Forests:
            for tree in forest.estimators_:
                self.mergedForest.estimators_.append(tree)
                self.n_estimators += 1

        self.n_rf = len(input_Forests)
        exampleForest = input_Forests[0]

        self.mergedForest.n_estimators = len(self.mergedForest.estimators_)
        self.mergedForest.n_classes_ = exampleForest.n_classes_
        self.mergedForest.n_outputs_ = exampleForest.n_outputs_
        self.mergedForest.classes_ = exampleForest.classes_

        return self.mergedForest

    def mergeAccuracy(self, input_Forests, n_trees_to_merge, x_valid, y_valid):
        params = input_Forests[0].get_params()
        self.mergedForest = RandomForestClassifier(**params)
        self.mergedForest.estimators_ = []

        for forest in input_Forests:
            predictions = []
            for tree in forest.estimators_:
                predictions.append(tree.predict(x_valid))
            accuracies = []
            for pred in predictions:
                accuracies.append(accuracy_score(y_valid, pred))
            best_trees_idx = sorted(range(len(accuracies)), key=lambda k: accuracies[k], reverse=True)[:n_trees_to_merge]
            best_trees = [forest.estimators_[i] for i in best_trees_idx]
            for tree in best_trees:
                self.mergedForest.estimators_.append(tree)
                self.n_estimators += 1

        self.n_rf = len(input_Forests)
        exampleForest = input_Forests[0]
        self.mergedForest.n_estimators = len(self.mergedForest.estimators_)
        self.mergedForest.n_classes_ = exampleForest.n_classes_
        self.mergedForest.n_outputs_ = exampleForest.n_outputs_
        self.mergedForest.classes_ = exampleForest.classes_

        return self.mergedForest

class FL_DP_Forest:
    """A smaller ε will yield better privacy but less accurate response. Small values of ε require to provide very similar outputs when given similar inputs,
    and therefore provide higher levels of privacy; large values of ε allow less similarity in the outputs, and therefore provide less privacy"""
    def __init__(self):
        self.n_rf = 0
        self.n_estimators = 0
    def mergeALL(self, input_Forests):
        params = input_Forests[0].get_params()
        self.mergedForest = DP_RandomForestClassifier(**params)
        self.mergedForest.estimators_ = []

        for forest in input_Forests:
            for tree in forest.estimators_:
                self.mergedForest.estimators_.append(tree)
                self.n_estimators += 1

        self.n_rf = len(input_Forests)
        exampleForest = input_Forests[0]

        self.mergedForest.n_estimators = len(self.mergedForest.estimators_)
        self.mergedForest.n_classes_ = exampleForest.n_classes_
        self.mergedForest.n_outputs_ = exampleForest.n_outputs_
        self.mergedForest.classes_ = exampleForest.classes_

        return self.mergedForest

    def mergeAccuracy(self, input_Forests, n_trees_to_merge, x_valid, y_valid):
        params = input_Forests[0].get_params()
        self.mergedForest = DP_RandomForestClassifier(**params)
        self.mergedForest.estimators_ = []

        for forest in input_Forests:
            predictions = []
            for tree in forest.estimators_:
                predictions.append(tree.predict(x_valid))
            accuracies = []
            for pred in predictions:
                accuracies.append(accuracy_score(y_valid, pred))
            best_trees_idx = sorted(range(len(accuracies)), key=lambda k: accuracies[k], reverse=True)[:n_trees_to_merge]
            best_trees = [forest.estimators_[i] for i in best_trees_idx]
            for tree in best_trees:
                self.mergedForest.estimators_.append(tree)
                self.n_estimators += 1

        self.n_rf = len(input_Forests)
        exampleForest = input_Forests[0]
        self.mergedForest.n_estimators = len(self.mergedForest.estimators_)
        self.mergedForest.n_classes_ = exampleForest.n_classes_
        self.mergedForest.n_outputs_ = exampleForest.n_outputs_
        self.mergedForest.classes_ = exampleForest.classes_

        return self.mergedForest