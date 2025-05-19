import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.15,
            (1, 1): 0.55
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.075,
            (0, 0, 1): 0.075,
            (0, 1, 0): 0.075,
            (0, 1, 1): 0.075,
            (1, 0, 0): 0.075,
            (1, 0, 1): 0.075,
            (1, 1, 0): 0.275,
            (1, 1, 1): 0.275,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for x in [0, 1]:
            for y in [0, 1]:
                if abs(X_Y[(x, y)] - X[x] * Y[y]) > 1e-10:
                    return True
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndependent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for x in [0, 1]:
            for y in [0, 1]:
                for c in [0, 1]:
                    # Check if P(X,Y|C) = P(X|C)P(Y|C)
                    # This is equivalent to P(X,Y,C)*P(C) = P(X,C)*P(Y,C)
                    left_side = X_Y_C[(x, y, c)] * C[c]
                    right_side = X_C[(x, c)] * Y_C[(y, c)]
                    if abs(left_side - right_side) > 1e-10:
                        return False
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # The PMF for Poisson is: P(X=k) = (λ^k * e^(-λ)) / k!
    # Taking log: log(P(X=k)) = k*log(λ) - λ - log(k!)
    
    # Handle edge cases
    if rate <= 0:
        return float('-inf')  # Log of zero is negative infinity
        
    log_p = k * np.log(rate) - rate - np.sum([np.log(i) for i in range(1, k+1)]) if k > 0 else -rate
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    likelihoods = np.zeros(len(rates))
    
    for i, rate in enumerate(rates):
        # Calculate log-likelihood for this rate across all samples
        log_likelihood = 0
        for sample in samples:
            log_likelihood += poisson_log_pmf(sample, rate)
        likelihoods[i] = log_likelihood
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Find the index of the maximum log-likelihood
    max_likelihood_index = np.argmax(likelihoods)
    
    # Return the corresponding rate
    rate = rates[max_likelihood_index]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # For Poisson distribution, the MLE is the sample mean
    mean = np.mean(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Normal PDF formula: f(x) = (1 / (σ * √(2π))) * exp(-(x - μ)²/(2σ²))
    if std <= 0:
        raise ValueError("Standard deviation must be positive")
        
    # Calculate the coefficient term
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    
    # Calculate the exponent term
    exponent = -0.5 * ((x - mean) / std) ** 2
    
    # Calculate the full PDF
    p = coefficient * np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Filter data for the specified class
        class_data = dataset[dataset[:, -1] == class_value]
        
        # Remove the class column
        features = class_data[:, :-1]
        
        # Calculate mean and std for each feature
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        # Store the total number of samples and class samples for prior calculation
        self.total_samples = len(dataset)
        self.class_samples = len(class_data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Calculate likelihood as product of normal PDFs for each feature
        likelihood = 1.0
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # P(class|x) ∝ P(x|class) * P(class)
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get posterior probabilities for both classes
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        
        # Return class with higher posterior probability
        pred = 0 if posterior0 >= posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Get features and true labels
    features = test_set[:, :-1]
    true_labels = test_set[:, -1]
    
    # Get predictions for each instance
    predictions = []
    for x in features:
        try:
            pred = map_classifier.predict(x)
            predictions.append(pred)
        except TypeError:
            # If prediction fails, use a default class (0)
            predictions.append(0)
    
    predictions = np.array(predictions)
    
    # Calculate accuracy
    correct_predictions = np.sum(predictions == true_labels)
    acc = correct_predictions / len(test_set)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Convert inputs to numpy arrays
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)
    
    # Get dimensions
    d = len(x)
    
    # Check if covariance matrix is positive definite
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError("Covariance matrix must be positive definite")
    
    # Calculate the coefficient term
    coefficient = 1 / ((2 * np.pi) ** (d/2) * np.sqrt(np.linalg.det(cov)))
    
    # Calculate the exponent term
    diff = x - mean
    exponent = -0.5 * diff.T @ np.linalg.inv(cov) @ diff
    
    # Calculate the full PDF
    pdf = coefficient * np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Filter data for the specified class
        class_data = dataset[dataset[:, -1] == class_value]
        
        # Remove the class column
        features = class_data[:, :-1]
        
        # Calculate mean and covariance matrix
        self.mean = np.mean(features, axis=0)
        self.cov = np.cov(features, rowvar=False)
        
        # Store the total number of samples and class samples for prior calculation
        self.total_samples = len(dataset)
        self.class_samples = len(class_data)
        
        # Store class value
        self.class_value = class_value
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Calculate likelihood using the multivariate normal PDF
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # P(class|x) ∝ P(x|class) * P(class)
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get prior probabilities for both classes
        prior0 = self.ccd0.get_prior()
        prior1 = self.ccd1.get_prior()
        
        # Return class with higher prior probability
        pred = 0 if prior0 >= prior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get likelihood probabilities for both classes
        likelihood0 = self.ccd0.get_instance_likelihood(x)
        likelihood1 = self.ccd1.get_instance_likelihood(x)
        
        # Return class with higher likelihood probability
        pred = 0 if likelihood0 >= likelihood1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Store dataset info
        self.total_samples = len(dataset)
        self.class_value = class_value
        
        # Filter data for the specified class
        class_data = dataset[dataset[:, -1] == class_value]
        self.class_samples = len(class_data)
        
        # Remove the class column
        features = class_data[:, :-1]
        all_features = dataset[:, :-1]
        
        # Number of features
        self.n_features = features.shape[1]
        
        # Calculate the count of each value for each feature
        self.feature_probs = {}
        for feature_idx in range(self.n_features):
            feature_values = {}
            
            # Find all unique values for this feature in the entire dataset
            unique_values = np.unique(all_features[:, feature_idx])
            
            # Count occurrences of each value in class data
            for value in unique_values:
                # Count with Laplace smoothing (add 1 to each count)
                count = np.sum(features[:, feature_idx] == value) + 1
                # Probability with Laplace smoothing
                prob = count / (self.class_samples + len(unique_values))
                feature_values[value] = prob
                
            self.feature_probs[feature_idx] = feature_values
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Calculate likelihood as product of probabilities for each feature
        likelihood = 1.0
        
        for feature_idx in range(self.n_features):
            # Get value of current feature
            feature_value = x[feature_idx]
            
            # Get probability for this feature value
            if feature_value in self.feature_probs[feature_idx]:
                prob = self.feature_probs[feature_idx][feature_value]
            else:
                # If value wasn't seen during training, use epsilon
                prob = EPSILLON
                
            likelihood *= prob
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # P(class|x) ∝ P(x|class) * P(class)
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get posterior probabilities for both classes
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        
        # Return class with higher posterior probability
        pred = 0 if posterior0 >= posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get features and true labels
        features = test_set[:, :-1]
        true_labels = test_set[:, -1]
        
        # Get predictions for each instance
        predictions = []
        for x in features:
            try:
                pred = self.predict(x)
                predictions.append(pred)
            except (TypeError, ValueError) as e:
                # If prediction fails, use a default class (0)
                predictions.append(0)
        
        predictions = np.array(predictions)
        
        # Calculate accuracy
        correct_predictions = np.sum(predictions == true_labels)
        acc = correct_predictions / len(test_set)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


