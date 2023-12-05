import numpy as np
from decision_tree import DecisionTree

class AdaBoostClassifier():
    """
    AdaBoost Classifier Model
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    The algorithm used in this class is SAMME algorithm with bootstrapping instead of weights
    """
    
    def __init__(self, n_base_learner=10, seed=0) -> None:
        """
        Initialize the object with the hyperparameters
        n_base_learner: # of base learnes in the model (base learners are DecisionTree with max_depth=1)
        """
        self.n_base_learner = n_base_learner
        self.seed = seed

    def _calculate_amount_of_say(self, base_learner, X, y, sample_weights: np.array) -> float:
        """calculates the amount of say (see SAMME)"""
        K = np.unique(y).shape[0]
        preds = base_learner.predict(X)
        err = 1 - np.sum((preds == y) * sample_weights) / np.sum(sample_weights)
        amount_of_say = np.log((1-err)/err) + np.log(K-1)
        return amount_of_say

    def _fit_base_learner(self, X: np.array, y: np.array, sample_weights: np.array) -> DecisionTree:
        """Trains a Decision Tree model with depth 1 and returns the model"""
        base_learner = DecisionTree(max_depth=1)
        base_learner.train(X, y, sample_weights)
        base_learner.amount_of_say = self._calculate_amount_of_say(base_learner, X, y, sample_weights)

        return base_learner

    def _update_sample_weights(self, base_learner, X, y, sample_weights) -> np.array:
        """Updates sample weights (see SAMME)"""
        preds = base_learner.predict(X)
        matches = (preds == y)
        not_matches = (~matches).astype(int)
        new_sample_weights = sample_weights * np.exp(base_learner.amount_of_say*not_matches)
        # Normalize weights
        new_sample_weights = new_sample_weights / np.sum(new_sample_weights)

        return new_sample_weights
    
    def train(self, X_train: np.array, y_train: np.array) -> None:
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
            base_learner = self._fit_base_learner(X, y, sample_weights)
            self.base_learner_list.append(base_learner)
            sample_weights = self._update_sample_weights(base_learner, X, y, sample_weights)

    def _predict_proba_w_base_learners(self,  X: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        pred_prob_list = []
        pred_prob_weighted_list = []
        for base_learner in self.base_learner_list:
            pred_probs = base_learner.predict_proba(X)
            pred_prob_list.append(pred_probs)
            pred_prob_weighted_list.append(pred_probs*base_learner.amount_of_say)

        return pred_prob_list, pred_prob_weighted_list

    def predict_proba(self, X) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = []
        _, base_learners_pred_probs_weighted = self._predict_proba_w_base_learners(X)

        # Take the weighted sum of the predicted probabilities of base learners
        for obs in range(X.shape[0]):
            base_learner_weighted_probs_for_obs = [a[obs] for a in base_learners_pred_probs_weighted]
            # Calculate the average for each index
            obs_pred_probs = np.sum(base_learner_weighted_probs_for_obs, axis=0)
            pred_probs.append(obs_pred_probs)
        
        return pred_probs

    def predict(self, X) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds        