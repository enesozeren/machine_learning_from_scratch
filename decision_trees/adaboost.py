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
        preds = base_learner.predict(X)
        matches = (preds == y)
        not_matches = (~matches).astype(int)
        new_sample_weights = sample_weights * np.exp(base_learner.amount_of_say*not_matches)
        new_sample_weights = new_sample_weights / np.sum(new_sample_weights)

        return new_sample_weights

    def _update_dataset(self, X, y, sample_weights) -> tuple:
        """Creates bootstrapped samples w.r.t. sample weights"""
        np.random.seed(self.seed)
        n_samples = len(X)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        X_bootstrapped = X[bootstrap_indices]
        y_bootstrapped = y[bootstrap_indices]

        # Initialize sample weights equally back
        sample_weights = np.full(shape=n_samples, fill_value=1.0/n_samples)        

        return X_bootstrapped, y_bootstrapped, sample_weights

    def _update_base_learner_amount_of_say(self, base_learner_list):
        """Unit Normalization for amount of says of base learners"""
        amount_of_say_list = []

        for i in range(len(base_learner_list)):
            amount_of_say_list.append(base_learner_list[i].amount_of_say)

        updated_amount_of_say_list = amount_of_say_list / np.sum(amount_of_say_list)

        for i in range(len(base_learner_list)):
            base_learner_list[i].amount_of_say = updated_amount_of_say_list[i]

        return base_learner_list
    
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
            X, y, sample_weights = self._update_dataset(X, y, sample_weights)
        
        self.base_learner_list = self._update_base_learner_amount_of_say(self.base_learner_list)

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