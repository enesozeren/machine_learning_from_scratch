import numpy as np
from decision_tree import DecisionTree

class AdaBoostClassifier():
    """
    AdaBoost Classifier Model
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    The algorithm used in this class is SAMME algorithm with bootstrapping instead of weights
    """
    
    def __init__(self, n_base_learner=10) -> None:
        """
        Initialize the object with the hyperparameters
        n_base_learner: # of base learnes in the model (base learners are DecisionTree with max_depth=1)
        """
        self.n_base_learner = n_base_learner

    def _calculate_amount_of_say(self, base_learner, X, y) -> float:
        """calculates the amount of say (see SAMME)"""
        K = np.unique(y).shape[0]
        preds = base_learner.predict(X)
        err = 1 - np.sum(preds==y) / preds.shape[0]
        amount_of_say = np.log((1-err)/err) + np.log(K-1)
        return amount_of_say

    def _fit_base_learner(self, X, y) -> DecisionTree:
        """Trains a Decision Tree model with depth 1 and returns the model"""
        base_learner = DecisionTree(max_depth=1)
        base_learner.train(X, y)
        base_learner.amount_of_say = self._calculate_amount_of_say(base_learner, X, y)

        return base_learner

    def _update_sample_weights(self, base_learner, X, y, sample_weights) -> np.array:
        """Updates sample weights (see SAMME)"""
        pass

    def _update_dataset(self, X, y, sample_weights, m) -> tuple:
        pass
    
    def train(self, X_train, y_train) -> None:
        """
        trains base learners with given feature and label dataset 
        """
        X = X_train
        y = y_train

        # Initialize equal sample weights
        m = X.shape[0]
        sample_weights = np.full(shape=m, fill_value=1.0/m)
        self.base_learner_list = []
        for i in range(self.n_base_learner):
            base_learner = self._fit_base_learner(X, y)
            self.base_learner_list.append(base_learner)
            sample_weights = self._update_sample_weights(base_learner, X, y, sample_weights)
            X, y = self._update_dataset(X, y, sample_weights, m)

    def predict_proba(self) -> np.array:
        pass

    def predict(self) -> np.array:
        pass