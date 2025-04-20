###### Your ID ######
# ID1: 201337151
# ID2: 307854505
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    if X.max() != X.min():
        X = (X - X.mean()) / (X.max() - X.min())
    if y.max() != y.min():
        y = (y - y.mean()) / (y.max() - y.min())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    X = np.c_[np.ones(X.shape[0]), X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    J = np.sum((X.dot(theta) - y)**2) / (2 * len(y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(y)  # number of training examples
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    window_size = 10  # Look at last 10 iterations for trend
    increase_threshold = 2.0  # Allow up to 100% increase from minimum
    
    for i in range(num_iters):
        # Compute predictions
        h = X.dot(theta)
        # Compute gradients
        gradients = X.T.dot(h - y) / m
        # Update parameters
        theta = theta - alpha * gradients
        # Save the cost
        current_cost = compute_cost(X, y, theta)
        J_history.append(current_cost)
        
        # More robust early stopping:
        # 1. Only start checking after window_size iterations
        # 2. Compare against minimum in recent window
        # 3. Stop if cost is significantly higher than recent minimum
        if i >= window_size:
            recent_min = min(J_history[-window_size:])
            if current_cost > recent_min * increase_threshold:
                print(f"Warning: Cost increased significantly at iteration {i}. Stopping early.")
                break
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
   
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################

    # Compute the transpose of X
    X_T = X.T
    # Compute (X^T * X)
    XTX = X_T.dot(X)
    # Compute the inverse of (X^T * X)
    XTX_inv = np.linalg.inv(XTX)
    # Compute the optimal parameters: (X^T * X)^(-1) * X^T * y
    pinv_theta = XTX_inv.dot(X_T).dot(y)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()# optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(y)  # number of training examples
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    window_size = 10  # Look at last 10 iterations for trend
    divergence_threshold = 5.0  # Stop if cost is 5x higher than minimum seen
    min_cost_seen = float('inf')
    
    for i in range(num_iters):
        # Compute predictions
        h = X.dot(theta)
        # Compute gradients
        gradients = X.T.dot(h - y) / m
        # Update parameters
        theta = theta - alpha * gradients
        # Save the cost
        current_cost = compute_cost(X, y, theta)
        J_history.append(current_cost)
        
        # Update minimum cost seen
        min_cost_seen = min(min_cost_seen, current_cost)
        
        # Check for divergence
        if current_cost > min_cost_seen * divergence_threshold:
            return theta, J_history
            
        # Stop if improvement is small enough over window
        if i >= window_size:
            recent_improvement = J_history[-window_size] - current_cost
            if abs(recent_improvement) < 1e-8:
                break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    for alpha in alphas:
        theta_init = np.zeros(X_train.shape[1])
        theta_learned,_ = efficient_gradient_descent(X_train, y_train, theta_init, alpha, iterations)
        loss = compute_cost(X_val, y_val, theta_learned)
        ##if not np.isnan(loss) and not np.isinf(loss):
        alpha_dict[alpha] = loss
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    
    while len(selected_features) < 5:
        lowest_error = float('inf')
        next_best_feature = None
        
        # Calculate some values once outside the feature loop
        current_features = len(selected_features)
        theta_size = current_features + 2  # +2 for bias and the new feature we'll try
        
        # Try each remaining feature
        remaining_features = [f for f in range(X_train.shape[1]) if f not in selected_features]
        
        for feature in remaining_features:
            # Prepare feature matrix with selected features + current candidate
            features_to_try = selected_features + [feature]
            X_train_subset = X_train[:, features_to_try]
            X_val_subset = X_val[:, features_to_try]
            
            # Add bias term
            X_train_subset = apply_bias_trick(X_train_subset)
            X_val_subset = apply_bias_trick(X_val_subset)
            
            # Initialize theta - use zeros as they work well with normalized data
            theta = np.zeros(theta_size)
            
            # Train with reduced iterations
            theta, _ = efficient_gradient_descent(X_train_subset, y_train, theta, 
                                               best_alpha, iterations)
            
            # Compute validation error
            error = compute_cost(X_val_subset, y_val, theta)
            
            if error < lowest_error:
                lowest_error = error
                next_best_feature = feature
        
        selected_features.append(next_best_feature)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    # Get list of original column names
    original_features = list(df.columns)
    
    # Create a dictionary to store all new features
    new_features = {}
    
    # Step 1: Create squared features
    for feature in original_features:
        new_column_name = feature + "^2"
        new_features[new_column_name] = df[feature] * df[feature]
    
    # Step 2: Create interaction features
    for i in range(len(original_features)):
        for j in range(i + 1, len(original_features)):
            feature1 = original_features[i]
            feature2 = original_features[j]
            new_column_name = feature1 + "*" + feature2
            new_features[new_column_name] = df[feature1] * df[feature2]
    
    # Combine original dataframe with new features all at once
    df_poly = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly