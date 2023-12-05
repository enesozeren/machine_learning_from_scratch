import numpy as np
from collections import Counter
from treenode import TreeNode


class DecisionTree():
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1, 
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None) -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(# of features) features considered, 
                                                            if log then log(# of features) features considered
                                                            else all features are considered
        amount_of_say: (float) -> used for Adaboost algorithm                                                    
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def _class_probabilities(self, labels: list, sample_weights: np.array) -> list:
        label_counts = {}
        total_weight = sum(sample_weights)

        for label, weight in zip(labels, sample_weights):
            if label in label_counts: label_counts[label] += weight
            else: label_counts[label] = weight

        return [count / total_weight for count in label_counts.values()]

    def _data_entropy(self, labels: list, sample_weights: np.array) -> float:
        return self._entropy(self._class_probabilities(labels, sample_weights))
    
    def _partition_entropy(self, subset_1: np.array, subset_2: np.array, 
                           sample_weights_1: np.array, sample_weights_2:np.array) -> float:
        """
            subset_1: first group of samples
            subset_2: second group of samples
            sample_weights_1: first group sample weights
            sample_weights_2: secong group sample weights
        """
        total_count = subset_1.shape[0] + subset_2.shape[0]
        subset_1_entropy_contr = self._data_entropy(subset_1, sample_weights_1) * (subset_1.shape[0] / total_count)
        subset_2_entropy_contr = self._data_entropy(subset_2, sample_weights_2) * (subset_2.shape[0] / total_count)
        return subset_1_entropy_contr + subset_2_entropy_contr
    
    def _split(self, data: np.array, sample_weights: np.array, feature_idx: int, feature_val: float) -> tuple:
        
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        sample_weights_1 = sample_weights[mask_below_threshold]
        sample_weights_2 = sample_weights[~mask_below_threshold]

        return group1, group2, sample_weights_1, sample_weights_2
    
    def _select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects the features to use while splitting w.r.t. hyperparameter numb_of_features_splitting
        """
        feature_idx = list(range(data.shape[1]-1))

        if self.numb_of_features_splitting == "sqrt":
            fature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            fature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            fature_idx_to_use = feature_idx

        return fature_idx_to_use
        
    def _find_best_split(self, data: np.array, sample_weights: np.array) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups
        """
        min_part_entropy = 1e9
        fature_idx_to_use =  self._select_features_to_use(data)

        for idx in fature_idx_to_use:
            feature_val = np.median(data[:, idx])
            g1, g2, sw1, sw2 = self._split(data, sample_weights, idx, feature_val)
            part_entropy = self._partition_entropy(g1[:, -1], g2[:, -1], sw1, sw2)
            if part_entropy < min_part_entropy:
                min_part_entropy = part_entropy
                min_entropy_feature_idx = idx
                min_entropy_feature_val = feature_val
                g1_min, g2_min = g1, g2
                sw1_min, sw2_min = sw1, sw2

        return g1_min, g2_min, sw1_min, sw2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, sample_weights: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # Find best split
        split_1_data, split_2_data, sample_weights_1, sample_weights_2, \
            split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data, sample_weights)
        
        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # Calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, sample_weights_1, current_depth)
        node.right = self._create_tree(split_2_data, sample_weights_2, current_depth)
        
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array, sample_weights: np.array = None) -> None:
        """
        Trains the model with given X and Y datasets
        sample_weights: weight for each sample (used for AdaBoost algorithm), if none then every sample is equally weighted
        """
        
        if sample_weights==None:
            sample_weights=np.full(X_train.shape[0], fill_value=1.0/X_train.shape[0])

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, sample_weights=sample_weights, current_depth=0)

        # Calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        # Normalize the feature importance values
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)         