import numpy as np
from decision_tree import DecisionTree

class AdaBoostClassifier():
    """
    AdaBoost Classifier Model
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    The algorithm used in this class is SAMME algorithm with boosting with resampling
    """
    
    def __init__(self, n_base_learner=10) -> None:
        """
        Initialize the object with the hyperparameters
        n_base_learner: # of base learners in the model (base learners are DecisionTree with max_depth=1)
        """
        self.n_base_learner = n_base_learner

    def _calculate_amount_of_say(self, base_learner: DecisionTree, X: np.array, y: np.array) -> float:
        """calculates the amount of say (see SAMME)"""
        K = self.label_count
        preds = base_learner.predict(X)
        err = 1 - np.sum(preds==y) / preds.shape[0]
        amount_of_say = np.log((1-err)/err) + np.log(K-1)
        return amount_of_say

    def _fit_base_learner(self, X_bootstrapped: np.array, y_bootstrapped: np.array) -> DecisionTree:
        """Trains a Decision Tree model with depth 1 and returns the model"""
        base_learner = DecisionTree(max_depth=1)
        base_learner.train(X_bootstrapped, y_bootstrapped)
        base_learner.amount_of_say = self._calculate_amount_of_say(base_learner, self.X_train, self.y_train)

        return base_learner

    def _update_dataset(self, sample_weights: np.array) -> tuple:
        """Creates bootstrapped samples w.r.t. sample weights"""
        n_samples = self.X_train.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        X_bootstrapped = self.X_train[bootstrap_indices]
        y_bootstrapped = self.y_train[bootstrap_indices] 

        return X_bootstrapped, y_bootstrapped    

    def _calculate_sample_weights(self, base_learner: DecisionTree) -> np.array:
        """Calculates sample weights (see SAMME)"""
        preds = base_learner.predict(self.X_train)
        matches = (preds == self.y_train)
        not_matches = (~matches).astype(int)
        sample_weights = 1/self.X_train.shape[0] * np.exp(base_learner.amount_of_say*not_matches)
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)

        return sample_weights
    
    def train(self, X_train: np.array, y_train: np.array) -> None:
        """
        trains base learners with given feature and label dataset 
        """
        self.X_train = X_train
        self.y_train = y_train
        X_bootstrapped = X_train
        y_bootstrapped = y_train
        self.label_count = len(np.unique(y_train))

        self.base_learner_list = []   
        for i in range(self.n_base_learner):
            base_learner = self._fit_base_learner(X_bootstrapped, y_bootstrapped)
            self.base_learner_list.append(base_learner)
            sample_weights = self._calculate_sample_weights(base_learner)
            X_bootstrapped, y_bootstrapped = self._update_dataset(sample_weights)

    def _predict_scores_w_base_learners(self,  X: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        pred_scores = np.zeros(shape=(self.n_base_learner, X.shape[0], self.label_count))
        for idx, base_learner in enumerate(self.base_learner_list):
            pred_probs = base_learner.predict_proba(X)
            pred_scores[idx] = pred_probs*base_learner.amount_of_say

        return pred_scores

    def predict_proba(self, X: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = []
        base_learners_pred_scores = self._predict_scores_w_base_learners(X)

        # Take the avg scores and turn them to probabilities
        avg_base_learners_pred_scores = np.mean(base_learners_pred_scores, axis=0)
        column_sums = np.sum(avg_base_learners_pred_scores, axis=1)
        pred_probs = avg_base_learners_pred_scores / column_sums[:, np.newaxis]     
        
        return pred_probs

    def predict(self, X: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds        