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

    def train(self):
        pass

    def predict_proba(self):
        pass

    def predict(self):
        pass