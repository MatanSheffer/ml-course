import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any array-like where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    # extract labels (last column)
    labels = np.array(data)[:, -1]
    # count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)
    # convert to probabilities
    probs = counts / counts.sum()
    # gini impurity formula
    gini = 1.0 - np.sum(probs**2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any array-like where the last column holds the labels.

    Returns:
    - entropy: The entropy value (bits, log base 2).
    """
    labels = np.array(data)[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    # avoid log2(0) by masking zero probabilities
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
    return entropy


import numpy as np

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.impurity_func = impurity_func
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this node's children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels = np.array(self.data)[:, -1]
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance and store in self.feature_importance.

        FI = (|S| / |S_total|) * (InformationGain or GainRatio)
        """
        gain, _ = self.goodness_of_split(self.feature)
        self.feature_importance = (self.data.shape[0] / n_total_sample) * gain

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Returns:
        - goodness: the information gain or gain ratio
        - groups: dict mapping feature_value -> data_subset
        """
        data = np.array(self.data)
        total = data.shape[0]
        base_impurity = self.impurity_func(data)

        weighted_impurity = 0.0
        split_info = 0.0
        groups = {}

        for val in np.unique(data[:, feature]):
            subset = data[data[:, feature] == val]
            groups[val] = subset
            p = subset.shape[0] / total
            weighted_impurity += p * self.impurity_func(subset)
            if self.gain_ratio and p > 0:
                split_info -= p * np.log2(p)

        info_gain = base_impurity - weighted_impurity

        if self.gain_ratio:
            goodness = info_gain / split_info if split_info > 0 else 0.0
        else:
            goodness = info_gain

        return goodness, groups

    def split(self):
        """
        Split the current node: choose best feature and create children.
        Supports max depth and stops when no gain.
        """
        if self.depth >= self.max_depth:
            self.terminal = True
            return

        data = np.array(self.data)
        n_features = data.shape[1] - 1

        best_gain = -np.inf
        best_feat = None
        best_groups = None

        for feat in range(n_features):
            gain, groups = self.goodness_of_split(feat)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_groups = groups

        if best_gain <= 0:
            self.terminal = True
            return

        self.feature = best_feat
        self.children = []
        self.children_values = []
        for val, subset in best_groups.items():
            child = DecisionNode(
                subset,
                self.impurity_func,
                feature=-1,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child, val)
        self.terminal = False

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # impurity function to use
        self.chi = chi
        self.max_depth = max_depth  # maximum allowed tree depth
        self.gain_ratio = gain_ratio
        self.root = None

    def build_tree(self):
        """
        Build the decision tree on self.data until pure leaves or no gain.
        """
        # initialize root node
        self.root = DecisionNode(
            self.data,
            self.impurity_func,
            feature=-1,
            depth=0,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )
        # recursively split nodes
        nodes = [self.root]
        while nodes:
            node = nodes.pop(0)
            node.split()
            if not node.terminal:
                nodes.extend(node.children)

    def predict(self, instance):
        """
        Predict the label for a single instance (last element is ignored).
        """
        node = self.root
        while node and not node.terminal:
            feat = node.feature
            val = instance[feat]
            if val in node.children_values:
                idx = node.children_values.index(val)
                node = node.children[idx]
            else:
                break
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Calculate accuracy (%) on dataset (last column are true labels).
        """
        data = np.array(dataset)
        X = data[:, :-1]
        y_true = data[:, -1]
        correct = 0
        for i in range(data.shape[0]):
            pred = self.predict(data[i])
            if pred == y_true[i]:
                correct += 1
        return (correct / data.shape[0]) * 100

    def depth(self):
        return self.root.depth

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






