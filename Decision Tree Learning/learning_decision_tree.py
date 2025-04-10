import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns

class Node:
    def __init__(self, feature=None, val=None, result=None, true_branch=None, false_branch=None):
        self.feature=feature # integer representation of feature to split on
        self.val=val # val of feature to split on (ground truth)
        self.true_branch = true_branch # points to the node which corresponds to if the ground truth was true
        self.false_branch = false_branch # points to the node which corresponds to if the ground truth was false
        self.result = result # if the node is a decision/leaf, store the result (for example: 'Y'/'N', or 'unacc'/'acc'/etc...)

        self.feature_label = None # actual label of feature to split on
    def print_info(self):
        print(f'feature={self.feature}\nfeature label ={self.feature_label}\nval={self.val}\nresults={self.result}\ntrue_branch={self.true_branch}\nfalse_branch={self.false_branch}')


def train_test_split(df, test_size, train_size, random_state=None):
    """ Splits the input dataframe into 4 dataframes: X_test, y_test, X_train, and y_train.
    test_size and train_size = number btwn 0 and 1 representing size of split.
    Random state = optional seed to shuffle data. 
    Returns all classification labels and all feature labels too.
    returns (X_test, X_train, y_test, y_train, feature_labels, class_labels) """
    
    # SHUFFLE DATAFRAME
    df_shuff = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save feature labels
    feature_labels = df.columns[:-1]
    # Save classification labels
    class_labels = df[df.columns[-1]].unique()

    test_start = 0
    test_end = int(test_size*len(df))

    train_start = test_end
    train_end = int(train_size*len(df)+train_start)

    if train_end < len(df):
        train_end+=1


    X_test = df_shuff.iloc[test_start:test_end, :-1].to_numpy()
    X_train = df_shuff.iloc[train_start:train_end, :-1].to_numpy()

    y_test = df_shuff.iloc[test_start:test_end, -1:].to_numpy()
    y_train = df_shuff.iloc[train_start:train_end, -1:].to_numpy()



    return X_test, X_train, y_test, y_train, feature_labels.tolist(), class_labels.tolist()


def split_data(X, y, feature, val):
    """ splits dataset on feature according to the specified val """
    pos_dexes = np.where(X[:, feature] == val)[0]
    neg_dexes = np.where(X[:, feature] != val)[0]
    
    pos_X, pos_y = X[pos_dexes], y[pos_dexes] # positive examples
    neg_X, neg_y = X[neg_dexes], y[neg_dexes] # negative examples

    return pos_X, pos_y, neg_X, neg_y


def get_entropy(y):
    """ Returns entropy of y where y is a 2d numpy array of shape [n,1] (n rows, 1 column)"""
    unique_vals, counts = np.unique(y, return_counts=True) 
    N = len(y) # total number of items in y

    probabilities = counts/N

    entropy = np.sum( [-1*p*np.log2(p) for p in probabilities if p > 0] )
    return entropy

def id3_build_tree(X, y, feature_labels=None):
    """ Takes in data arrays X and y to build decision tree with id3 algorithm, which uses entropy to build tree by determining features w most to least discriminatory power. 
    Returns the root node of the decision tree """
    most_discrim_feature_and_val = None
    most_discrim_split_data = None
    initial_entropy = get_entropy(y)
    highest_information_gain = 0

    # GET INFORMATION GAIN FOR EACH FEATURE
    for feature in range(X.shape[1]):
        feature_vals = set(X[:, feature]) # make a set containing each value/quality the feature is capable of. 
        # GET ENTROPY FOR EACH FEATURE VALUE
        for feature_val in feature_vals:
        
            # Split the data along the feature value
            true_X, true_y, false_X, false_y = split_data(X, y, feature, feature_val) # note: ground truth is value == feature_value

            entropy_true_y = get_entropy(true_y)
            entropy_false_y = get_entropy(false_y)

            # Calculate probability of ground truth & probability of not ground truth
            prob_of_val = len(true_y) / len(y)
            prob_of_not_val = len(false_y) / len(y)

            # Information Gain = entropy_of_y - (P(ground_truth)*entropy(true)  + P(not_ground_truth)*entropy(false))
            information_gain = initial_entropy - ( (prob_of_val * entropy_true_y) + (prob_of_not_val * entropy_false_y) )
            #print(f'info_gain = {information_gain}, feature = {feature}, value = {feature_val}')
            # Update most discriminatory feature if needed
            if information_gain > highest_information_gain:
                highest_information_gain = information_gain
                most_discrim_feature_and_val = (feature, feature_val)
                most_discrim_split_data = true_X, true_y, false_X, false_y

    # CONTINUE BUILDING TREE FOR TRUE/FALSE BRANCHES
    if highest_information_gain > 0: 
        # BUILD TRUE/FALSE BRANCH
        true_X, true_y = most_discrim_split_data[0], most_discrim_split_data[1]
        false_X, false_y = most_discrim_split_data[2], most_discrim_split_data[3]
        true_branch = id3_build_tree(true_X, true_y, feature_labels=feature_labels)
        false_branch = id3_build_tree(false_X, false_y, feature_labels=feature_labels)
        feature_node = Node(feature = most_discrim_feature_and_val[0],val = most_discrim_feature_and_val[1], true_branch = true_branch, false_branch = false_branch)
        if type(feature_labels) != None:            
            feature_node.feature_label = feature_labels[feature_node.feature]
        return feature_node
    
    else:
        # If highest info gain == 0, then you have no features left to split on. In this case, set result to y[0][0].
        feature_node = Node(result=y[0][0])
        return feature_node


def make_decision(root, x):
    """ Makes decision for feature vector x based off of decision tree root.
    Returns the result of the decision """
    most_discrim_feature = root.feature
    current_node = root
    while current_node.feature != None:
        #current_node.print_info()
        #print()
        # If ground truth, then update to true branch
        if x[current_node.feature] == current_node.val:
            current_node = current_node.true_branch

        # If NOT ground truth, then update to false branch
        elif x[current_node.feature] != current_node.val:
            current_node = current_node.false_branch
    return current_node.result

def test_tree(root, X):
    """ Runs the decision tree root for every data point in dataset X, where X is a 2d array where X[i][feature] = value of x_i for that feature.
    Returns the decision made for each data point in the form of a 2d array with 1 column and x rows """
    results = []
    for x in X:
        results.append(make_decision(root, x))
    res_arr = np.array(results)
    res_arr = np.vstack(res_arr)
    return res_arr

def get_conf_matrix(true_labels, predictions, class_labels):
    """ Takes in the true labels of a test set, the predicted labels of a test set, and the classification labels where true_labels[i]'s classification label is class_label[i] 
    Returns the confusion matrix as a 2d numpy array """
    labels = None
    if type(class_labels) != list: # convert to a python list if of type pandas index or something weird like that. error handling.
        labels = list(class_labels)
    elif type(class_labels) == list:
        labels = class_labels
    conf_matrix = [[0]*len(labels) for i in range(len(labels))] # initialize this all to 0 but maybe do it in a numpy array
    true_labels = true_labels.flatten()
    predictions = predictions.flatten()
    for i in range(len(predictions)):
        pred_index = labels.index(predictions[i])
        true_index = labels.index(true_labels[i])
        conf_matrix[true_index][pred_index] += 1
    return np.array(conf_matrix, dtype=int)

def get_stats(conf_matrix, class_labels):
    """ Returns a tuple (1,2):
    1) a dataframe that for each class has the number of True Positives, False Positives, True Negatives, False Negatives, Precision, Recall, and F1 score. Also has columns for Macro Avg & Weighted Avg of each stat 
    2) The accuracy of the data (float)
    """
    
    num_preds = np.sum(conf_matrix) # total number of predictions made
    #print(size)
    col_labels = [class_label for class_label in class_labels]+['Macro_Avg','Weighted_Avg']
    index_labels = ['True_Positives','False_Positives','True_Negatives','False_Negatives', 'Precision', 'Recall', 'F1_Score']
    conf_arr = np.array(conf_matrix)

    
    # GET RESULTS
    stats_df = pd.DataFrame(columns=col_labels, index= index_labels)
    for i, true_label in enumerate(class_labels):
        
        TPs = conf_arr[i][i] # it was of this class, and was predicted as this class
        FPs = np.sum(conf_arr[i,:]) - TPs # it wasn't of this class and was predicted as this class
        #print(f'sum of column = {np.sum(conf_arr[:,i])}, sum of row = {np.sum(conf_arr[i,:])}')
        #print(conf_arr[i,:])
        #print(conf_arr[:,i])
        FNs = np.sum(conf_arr[:,i]) - TPs # it was of this class, but was predicted to not be
        TNs = num_preds - (TPs + FPs + FNs)


        if TPs+FPs != 0:
            precision = TPs / (TPs + FPs)
        else:
            precision = 0
        
        if TPs+FNs != 0:
            recall = TPs / (TPs + FNs)
        else:
            recall = 0

        if precision != 0 and recall != 0:
            f1 = 2 / ( (1/precision) + (1/recall) )
        elif precision == 0 and recall == 0:
            f1 = 0
        elif precision == 0:
            f1 = 2/(1/recall)
        elif recall == 0:
            f1 = 2/(1/precision)
        #print(f'{true_label} : precis = {precision}, recall={recall}, f1={f1}, TPs={TPs}, FPs={FPs}, FNs={FNs}')

        stats_df[true_label] = [TPs, FPs, TNs, FNs, precision, recall, f1]
    
    # Calculate Macro Average
    stats_df.loc[index_labels, 'Macro_Avg'] = stats_df.loc[index_labels].mean(axis=1)

    # Calculate Weigthed Average
    support = stats_df.loc['True_Positives'] + stats_df.loc['False_Negatives']
    weighted_avg = (stats_df.loc[index_labels] * support).sum(axis=1) / support.sum()
    stats_df.loc[index_labels, 'Weighted_Avg'] = weighted_avg.values
    
    # CALCULATE ACCURACY
    df = stats_df.drop(columns=['Macro_Avg', 'Weighted_Avg'])
    counts = df.sum(axis=1)
    correct_preds = counts['True_Positives']
    accuracy = correct_preds/num_preds

    return stats_df.astype(float), accuracy

def get_test_train_size(test_set, train_set):
    return len(test_set), len(train_set)

def iterative_id3_build(df, test_size, train_size, step_size, random_state=None, feature_labels=None):
    """
    Builds and tests decision tree with increasingly larger portion of training data. Returns a tuple of 2 arrays (1,2):
    1) fractions of the training data used to make predictions (x axis)
    2) accuracies of the tree from each fraction (y axis)
    """
    X_test, X_train, y_test, y_train, feature_labels, class_labels = train_test_split(df, test_size, train_size, random_state=random_state)
    fractions = []
    accuracies = []
    N = len(X_train)-1

    num_steps = N//step_size
    #print(f'step_size = {step_size}, num_steps = {num_steps}')


    for step in range(1,num_steps+1):
        fraction_of_set = step*step_size / N
        root = id3_build_tree(X_train[0:step*step_size], y_train[0:step*step_size], feature_labels=feature_labels)
        results = test_tree(root, X_test)
        conf_matrix = get_conf_matrix(y_test, results, class_labels)
        stats_df, accuracy = get_stats(conf_matrix, class_labels)
        fractions.append(fraction_of_set)
        accuracies.append(accuracy)

    if fraction_of_set != 1:
        # HANDLE FINAL RESULTS
        fraction_of_set = 1
        root = id3_build_tree(X_train, y_train, feature_labels=feature_labels)
        results = test_tree(root, X_test)
        conf_matrix = get_conf_matrix(y_test, results, class_labels)
        stats_df, accuracy = get_stats(conf_matrix, class_labels)
        fractions.append(fraction_of_set)
        accuracies.append(accuracy)
    
    return fractions, accuracies

def main(filename, train_size, test_size, show_plots=True):
    df = pd.read_csv(filename)

    # GET TRAIN/TEST SPLIT
    X_test, X_train, y_test, y_train, feature_labels, class_labels = train_test_split(df, test_size=test_size, train_size=train_size)

    # PRINT TRAIN/TEST SIZE
    test_size, train_size = get_test_train_size(X_test, X_train)
    print(f"TEST SIZE = {test_size}, TRAIN SIZE = {train_size}")

    # BUILD TREE
    root = id3_build_tree(X_train,y_train, feature_labels=feature_labels)

    # TEST TREE WITH TEST SET
    results = test_tree(root, X_test)

    # CREATE & SHOW CONFUSION MATRIX
    conf_matrix = get_conf_matrix(y_test, results, class_labels)
    if show_plots:
        plt.figure(figsize=(12,10))
        sns.heatmap(conf_matrix, cmap='RdBu', annot=True, xticklabels=class_labels, yticklabels=class_labels,fmt='g', vmin=0)
        plt.title('Confusion Matrix')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.autoscale()
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

    # GET STATISTICS
    stats_df, accuracy = get_stats(conf_matrix, class_labels)

    # PLOT RESULTS
    if show_plots:
        plt.figure(figsize=(12,10))
        new_df = stats_df.drop(index=['True_Positives', 'False_Positives', 'True_Negatives', 'False_Negatives'], inplace=False)
        sns.heatmap(new_df, cmap='RdBu', annot=True, vmin=0, fmt='g')
        plt.title('Precision, F1, and Recall')
        plt.xlabel("Class Label")
        plt.ylabel("Metric")
        plt.autoscale()
        plt.tight_layout()
        plt.savefig('Precision new.png')
        plt.show()
        
        plt.figure(figsize=(12,10))
        new_df2 = stats_df.drop(index=['Precision','F1_Score','Recall', 'True_Negatives', 'False_Negatives'], inplace=False)
        sns.heatmap(new_df2, cmap='RdBu', annot=True, vmin=0, fmt='g')
        plt.title('TPs and FPs')
        plt.xlabel("Class Label")
        plt.ylabel("Metric")
        plt.autoscale()
        plt.tight_layout()
        plt.savefig('TPs and FPs new.png')
        plt.show()

        plt.figure(figsize=(12,10))
        new_df3 = stats_df.drop(index=['Precision','F1_Score','Recall', 'True_Positives', 'False_Positives'], inplace=False)
        sns.heatmap(new_df3, cmap='RdBu', annot=True, vmin=0, fmt='g')
        plt.title('TNs and FNs')
        plt.xlabel("Class Label")
        plt.ylabel("Metric")
        plt.autoscale()
        plt.tight_layout()
        plt.savefig('TNs and FNs new.png')        
        plt.show()

        # PRINT ACCURACY & STATS
        print(f'TOTAL ACCURACY = {accuracy}')
        #print(f'\n{stats_df}\n')
    return stats_df, root, feature_labels

def plot_test_results(fractions, accuracies):
    x = np.array(fractions)
    y = np.array(accuracies)

    # convert fractions to percentages and round both arrays to 2 decimals
    x = (x*100).round(decimals=2)
    y = (y*100).round(decimals=2)

    plt.figure(figsize=(10,8))

    # Fit a trend line to the data
    trendline = np.polyfit(x, y, 9)
    trend = np.poly1d(trendline)
    plt.plot(x, trend(x), label='Trend line', color='red', alpha=0.6, linestyle='--')


    plt.scatter(x,y)
    plt.title("Learning Curve (accuracy as a factor of percentage of learning example)")
    plt.xlabel("Percentage of Training Set Used")
    plt.ylabel("Accuracy Percentage")
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.grid()
    plt.autoscale()
    plt.tight_layout()
    plt.legend()
    plt.savefig('Learning Curve')
    plt.show()


def print_tree(node, spacing=""):
    ''' Given the root node from a decision tree, prints the tree '''
    # Base case: if this is a leaf node, print the result
    if node.result is not None:
        print(spacing + f"Predict: {node.result}")
        return

    else:
        # Print the feature and value that this node splits on
        print(spacing + f"Is feature {node.feature_label} == {node.val}?")

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        print_tree(node.false_branch, spacing + "  ")

df = pd.read_csv('car.csv')
print(df.nunique())
df.head()
# Run on increasingly large sample size of training data
print("---Iterative Build---")
fractions, accuracies = iterative_id3_build(df, test_size = 0.8, train_size = 0.2, step_size = 8)
print(f'{len(fractions)} steps made')
plot_test_results(fractions,accuracies)


# Run on full training set
print('---Running on Full Training Set---')
stats_df, root, feature_labels = main('car.csv', train_size=0.8, test_size=0.2)
print_tree(root)