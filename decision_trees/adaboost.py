import numpy as np
from decision_tree import DecisionTree

class AdaBoostClassifier():
    """
    AdaBoost Classifier Model
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """
    
    def __init__(self, n_base_learner=10) -> None:
        """
        Initialize the object with the hyperparameters
        n_base_learner: # of base learnes in the model (base learners are DecisionTree with max_depth=1)
        """
        self.n_base_learner = n_base_learner

    def _fit_base_learner(self, X_train, y_train, sample_weights):
        pass

    def _update_sample_weights(self, base_learner, X_train, y_train, sample_weights):
        pass
    
    def train(self, X_train, y_train):
        """
        trains base learners with given feature and label dataset 
        """

        # Initialize equal sample weights
        m = X_train.shape[0]
        sample_weights = np.full(shape=m, fill_value=1.0/m)
        self.base_learner_list = []
        for i in range(self.n_base_learner):
            base_learner = self._fit_base_learner(X_train, y_train, sample_weights)
            self.base_learner_list.append(base_learner)
            sample_weights = self._update_sample_weights(base_learner, X_train, y_train, sample_weights)

    def predict_proba(self):
        pass

    def predict(self):
        pass