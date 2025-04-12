import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import time

def sigmoid(X):
    """ returns sigmoid of x where x is a single number """
    clipped_X = np.clip(X, -500, 500)  # Clip x to prevent overflow or underflow
    return np.reciprocal(1 + np.exp(np.negative(clipped_X)))

def convert_to_binary(Y):
    """ Converts to binary outputs for each node, where the max is 1 and everything else is 0. Each row is treated seperately. """
    binary_arr = []
    for y in Y:
        binary = np.zeros_like(y)
        max_idx = np.argmax(y)
        binary[max_idx] = 1
        binary_arr.append(np.array(binary, dtype=int))
    binary_arr = np.array(binary_arr, dtype=int)
    return binary_arr


class Layer:
  def __init__(self, input_dim, output_dim):
    self.in_dim = input_dim # Number of inputs
    self.out_dim = output_dim # Number of output nodes

    self.Inputs = None # Inputs to the layer (2d numpy array)
    self.Outputs = None # Outputs to the layer (2d numpy array)

    self.Weights = np.random.randn(self.in_dim, self.out_dim) # Set to random # btwn -1 and 1
    self.biases = np.random.randn(1, self.out_dim) # Set to random @ btwn -1 and 1
    self.Deltas = None # Set using either set_Case1_Deltas or set_Case2_Deltas

  def forward_prop(self, X):
    """ 
    Does forward propogation for input array X and sets outputs to self.Outputs 
    """
    Weighted_Sum_of_Inputs = X @ self.Weights + self.biases # Calculates weighted sum of inputs where X is the input array.
    self.Outputs = sigmoid(Weighted_Sum_of_Inputs) # Updates outputs using the activation function on the weighted sum


  def set_Case1_Deltas(self, Desired): # Case 1 = node is an output node
    """
    Updates deltas for case 2 when node is an internal node 
    """
    self.Deltas = self.Outputs * (1 - self.Outputs) * (Desired - self.Outputs)


  def set_Case2_Deltas(self, next_layer_Case1_Deltas, next_layer_Adjusted_Weights): # Case 2 = node is an internal node 
    """
    Updates deltas for case 2 when node is an internal node 
    
    """
    Weighted_Sum_of_Case1_Deltas = next_layer_Case1_Deltas @ next_layer_Adjusted_Weights.T
    self.Deltas = self.Outputs * (1 - self.Outputs) * Weighted_Sum_of_Case1_Deltas

  def do_update(self, learning_rate, previous_layer_Outputs):
    """
    Updates Weights and biases using formula w_ij' = w_ij + n*delta_j*w_i.
    """

    self.Weights = self.Weights + (learning_rate * (previous_layer_Outputs.T @ self.Deltas) / self.Deltas.shape[0])
    self.biases = self.biases + learning_rate * np.mean(self.Deltas, axis=0, keepdims=True)
    #print(type(learning_rate), type(previous_layer_Outputs), type(self.Deltas))

  def info(self, layer_name):
    info = f'''PRINTING {layer_name} INFO\ninput dims=\n{self.in_dim}\noutput dims=\n{self.out_dim}\nWeights=\n{self.Weights}\nbiases=\n{self.biases}\ndeltas=\n{self.Deltas}\n'''


class NeuralNetwork:
    def __init__(self, hidden_layer, output_layer):
      self.hidden_layer = hidden_layer
      self.output_layer = output_layer

      self.learning_rate = None

      self.Binary_Outputs = None

    def network_forward_prop(self, X):
      """ Does forward propogation for entire neural network. Does not set binary outputs. """
      self.hidden_layer.forward_prop(X)
      
      self.output_layer.forward_prop(self.hidden_layer.Outputs)

    def network_back_prop(self, X, Desired, learning_rate):
      """ Does back propogation for entire neural network """
      self.output_layer.set_Case1_Deltas(Desired) # Sets case 1 deltas for the output layer using the desired output to calculate the error.
      self.output_layer.do_update(learning_rate, self.hidden_layer.Outputs) # Updates the Weights and biases for the output layer using the learning rate and outputs from the hidden layer.

      self.hidden_layer.set_Case2_Deltas(self.output_layer.Deltas, self.output_layer.Weights) # Sets hidden layer's (case 2) deltas using output layer's (case 1) deltas and output layer's updated Weights.
      self.hidden_layer.do_update(learning_rate, X) # Updates the hidden layer's Weights and biases

    def make_predictions(self, X_test):
        """ Makes predictions for testing data """
        # Run forward pass on network with test data to update outputs
        self.network_forward_prop(X_test)
        self.Binary_Outputs = convert_to_binary(self.output_layer.Outputs)

    def get_prediction_stats(self, X_test, D_test):
        """ Returns the number of correct predictions """
        self.make_predictions(X_test)
        correct = np.sum(self.Binary_Outputs == D_test)
        return correct

    def info(self):
      self.hidden_layer.info('hidden layer')
      self.output_layer.info('output layer')
      print('Binary Outputs:\n', self.Binary_Outputs)

    def train(self, training_data, testing_data, learning_rate, num_epochs, minibatch_size):
        """ Trains the neural network using the training data, testing data and learning rate over num_epochs epochs with minibatch size of minibatch_size """
        
        self.learning_rate = learning_rate
        X_train, D_train = training_data
        X_test, D_test = testing_data
        stats = [0]

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")


            batch_num = 0
            for start_idx in range(0, len(X_train), minibatch_size): # this keeps index errors from happening
                end_idx = min(start_idx+minibatch_size, len(X_train))

                batch_num+=1 # keep track of batch number for bugfixing

                # Get slice of X and D for minibatch
                minibatch_X = X_train[start_idx:end_idx, :]
                minibatch_D = D_train[start_idx:end_idx, :]

                # Error handling
                minibatch_length = len(minibatch_X)
                if minibatch_length == 0:
                    break

                # Run forward pass on network with training data
                self.network_forward_prop(minibatch_X)

                # Back propogate the error through the network
                self.network_back_prop(minibatch_X, minibatch_D, self.learning_rate)
            # Test the network on the test set at the end of an epoch.
            stats.append(self.get_prediction_stats(X_test,D_test))
        return stats

def get_train_test_data(train_filename, test_filename, NOutput):
    
    # GET TRAINING DATA
    train_df = pd.read_csv(train_filename)
    train_arr = train_df.to_numpy()
    X_train = train_arr[:,1:] # The inputs are everything except for the first column of the array
    X_train = X_train/np.max(X_train) # Normalize inputs to be between 0 and 1
    y_train = train_arr[:,0] # Get desired outputs as the first column of the array

    # Convert y train to 1 hot encoding
    new_y_train = np.zeros((y_train.shape[0], NOutput))
    rows = np.arange(y_train.shape[0])
    cols = y_train.flatten()
    new_y_train[rows, cols] = 1

    training_data = (X_train, new_y_train)


    # GET TESTING DATA
    test_df = pd.read_csv(test_filename)
    test_arr = train_df.to_numpy()
    X_test = test_arr[:,1:]
    X_test = X_test/np.max(X_train) # Normalize inputs to be between 0 and 1
    y_test = test_arr[:, 0] # Get desired outputs as the last column of the array

    # Convert y test to 1 hot encoded format
    new_y_test = np.zeros((y_test.shape[0], NOutput))
    rows = np.arange(y_test.shape[0])
    cols = y_test.flatten()
    new_y_test[rows, cols] = 1

    testing_data = (X_test, new_y_test)

    return training_data, testing_data

def test_2(training_data, testing_data):
  """ Tests effect of changing learning rate.
  Saves plot and CSV related to data obtained. """
  learning_rates = np.array([0.001, 0.01, 1.0, 10, 100])
  num_epochs = 30
  minibatch_size = 20

  test2_all_accuracies = []

  # Run test
  for learning_rate in learning_rates:
      print(f'---TRAINING FOR LEARNING RATE {learning_rate}---')
      hidden_layer = Layer(784, 30)
      output_layer = Layer(30, 10)
      nn = NeuralNetwork(hidden_layer, output_layer)

      accuracies = nn.train(training_data, testing_data, learning_rate, num_epochs, minibatch_size)
      test2_all_accuracies.append(np.array(accuracies))
  test2_all_accuracies = np.array(test2_all_accuracies)
  test2_all_accuracies = (test2_all_accuracies/(testing_data[1].shape[0]*testing_data[1].shape[1])) * 100
  
  # Put data in dataframe to save and also get max info
  lrs = [f'{lr}' for lr in learning_rates]
  test2 = pd.DataFrame(columns=lrs)
  for i, lr in enumerate(lrs):
      test2[lr] = test2_all_accuracies[i]
  test2.index.rename('Epoch', inplace=True)
  max_accuracy_index = test2.idxmax()
  max_accuracy = test2.max()

  # Print accuracy info
  print("Epoch of max accuracy % for each learning rate:")
  max_accuracy_index.rename('Epoch', inplace=True)
  max_accuracy_index.index.rename('Learning Rate', inplace=True)
  print(max_accuracy_index[1:])
  print("\nMax accuracy % for each learning rate:")  
  max_accuracy.rename('Accuracy (%)', inplace=True)
  max_accuracy.index.rename('Learning Rate', inplace=True)
  print(max_accuracy[1:])

  # Save results to CSV
  test2.to_csv('test2.csv')


  plt.close()
  plt.figure(figsize=(12,10))
  epochs = np.arange(0,31)

  # Define different line styles and colors
  line_styles = ['-', '--', '-.', ':', '-']
  colors = ['red', 'green', 'blue', 'orange', 'purple']
  plt.figure(figsize=(12, 8))

  # Plot results
  for lr, color, style in zip(list(test2.columns[1:]), colors, line_styles):
      x = test2.index.tolist()[1:]
      y = test2[str(lr)][1:]
      plt.plot(x, y, marker='o', color=color, linestyle=style, label=f'{lr}', alpha=0.6)

  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Accuracy vs. Epoch for Different Learning Rates')
  plt.legend()
  plt.grid()
  plt.autoscale()
  plt.tight_layout
  plt.savefig('test2 acc results.png')
  plt.show()

def test_3(train_data, test_data):
  """ Tests effect of changing minibatch size.
  Saves plots and CSVs related to data obtained. """
  learning_rate = 3
  num_epochs = 30
  minibatch_sizes = [1,5,20,100,300]
  test3_all_accuracies = []
  test3_times = []

  # Run test
  for minibatch_size in minibatch_sizes:
      print(f'---TRAINING FOR MINIBATCH SIZE {minibatch_size}---')
      start = time.time()
      hidden_layer = Layer(784, 30)
      output_layer = Layer(30, 10)
      nn = NeuralNetwork(hidden_layer, output_layer)
      accuracies = nn.train(training_data, testing_data, learning_rate, num_epochs, minibatch_size)
      test3_all_accuracies.append(np.array(accuracies))
      test3_times.append(time.time() - start)
      print(f'time = {time.time() - start}')
  test3_all_accuracies = np.array(test3_all_accuracies)
  test3_all_accuracies = (test3_all_accuracies/(testing_data[1].shape[0]*testing_data[1].shape[1])) * 100

  test3_times = np.array(test3_times)
  epochs = np.arange(0,31)

  # Put data in dataframe to save and also get max info
  mbs = [f'{mb}' for mb in minibatch_sizes]
  test3_acc = pd.DataFrame(columns=mbs)
    
  line_styles = ['-', '--', '-.', ':', '-']
  colors = ['red', 'green', 'blue', 'orange', 'purple']
  plt.figure(figsize=(12,8))

  # Plot accuracy results
  for mbs, color, style in zip(list(test3_acc.columns[1:]), colors, line_styles):
      x = test3_acc.index.tolist()[1:]
      y = test3_acc[str(mbs)][1:]
      plt.plot(x, y, marker='o', color=color, linestyle=style, label=f'{mbs}', alpha=0.6)

  # Find epoch of max accuracy % and max accuracy % for each minibatch size
  max_accuracy_index = test3_acc.idxmax()
  max_accuracy = test3_acc.max()

  print("Epoch of max accuracy % for each minibatch size rate:")
  max_accuracy_index.rename('Epoch', inplace=True)
  max_accuracy_index.index.rename('Minibatch Size', inplace=True)
  print(max_accuracy_index[1:])
  print("\nMax accuracy % for each minibatch size:")
  max_accuracy.rename('Accuracy (%)', inplace=True)
  max_accuracy.index.rename('Minibatch Size', inplace=True)
  print(max_accuracy[1:]) 

  # Print fastest and slowest time info, also  put into dataframe.
  t3_time_df = pd.DataFrame(columns=['Minibatch Size','Runtime (s)'])
  t3_time_df['Runtime (s)'] = np.array(test3_times)
  t3_time_df['Minibatch Size'] = np.array(minibatch_sizes)
  t3_time_df.set_index('Minibatch Size', inplace=True)
  

  # Find the index of the row with the fastest runtime
  fastest_index = df['Runtime (s)'].idxmin()
  # Find the index of the row with the slowest runtime
  slowest_index = df['Runtime (s)'].idxmax()

  fastest_runtime = df.loc[fastest_index, 'Runtime (s)']
  slowest_runtime = df.loc[slowest_index, 'Runtime (s)']

  # Extract the minibatch size for the fastest and slowest runtimes
  fastest_minibatch_size = df.loc[fastest_index, 'Minibatch Size']
  slowest_minibatch_size = df.loc[slowest_index, 'Minibatch Size']

  print("Minibatch size for the fastest runtime:", fastest_minibatch_size, '(runtime =',fastest_runtime, "seconds)")
  print("Minibatch size for the slowest runtime:", slowest_minibatch_size, '(runtime =',slowest_runtime, "seconds)")

  # Plot time results
  plt.close()
  plt.figure(figsize=(12,8))
  sns.barplot(data=t3_time_df, x='Minibatch Size', y='Runtime (s)')
  plt.title('Runtimes for Each Minibatch Size')
  plt.autoscale()
  plt.tight_layout()
  plt.savefig('minibatch times.png')
  plt.show()


  # Save results to CSV
  test3_acc.to_csv('test3_acc.csv')
  t3_time_df.to_csv('t3_time.csv')



  plt.close()

  plt.figure(figsize=(12,10))
  for i, minibatch_size in enumerate(minibatch_sizes):
      plt.plot(epochs, test3_all_accuracies[i], label=f'Minibatch Size = {minibatch_size}')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Accuracy vs. Epoch for Different Minibatch Sizes')
  plt.grid(True)
  plt.autoscale()
  plt.tight_layout
  plt.savefig('test3 acc results.png')
  plt.show()

  plt.figure(figsize=(12,10))
  sns.barplot(x=minibatch_sizes, y=test3_times)
  plt.title("Runtime for Each Minibatch Size")
  plt.xlabel("Minibatch Size")
  plt.ylabel("Runtime (s)")
  plt.autoscale()
  plt.tight_layout
  plt.savefig('test3 time results.png')
  plt.show()

def test_4(training_data, testing_data):
  """ Tests effect of changing learning rate.
  Saves plot and CSV related to data obtained. """
  learning_rates = np.array([1, 1.5, 2, 2.5, 3])
  num_epochs = 30
  minibatch_size = 20

  test4_all_accuracies = []

  # Run test
  for learning_rate in learning_rates:
      print(f'---TRAINING FOR LEARNING RATE {learning_rate}---')
      hidden_layer = Layer(784, 30)
      output_layer = Layer(30, 10)
      nn = NeuralNetwork(hidden_layer, output_layer)

      accuracies = nn.train(training_data, testing_data, learning_rate, num_epochs, minibatch_size)
      test4_all_accuracies.append(np.array(accuracies))
  test4_all_accuracies = np.array(test4_all_accuracies)
  test4_all_accuracies = (test4_all_accuracies/(testing_data[1].shape[0]*testing_data[1].shape[1])) * 100
  
  # Put data in dataframe to save and also get max info
  lrs = [f'{lr}' for lr in learning_rates]
  test4 = pd.DataFrame(columns=lrs)
  for i, lr in enumerate(lrs):
      test4[lr] = test4_all_accuracies[i]
  test4.index.rename('Epoch', inplace=True)
  max_accuracy_index = test4.idxmax()
  max_accuracy = test4.max()

  # Print accuracy info
  print("Epoch of max accuracy % for each learning rate:")
  max_accuracy_index.rename('Epoch', inplace=True)
  max_accuracy_index.index.rename('Learning Rate', inplace=True)
  print(max_accuracy_index[1:])
  print("\nMax accuracy % for each learning rate:")  
  max_accuracy.rename('Accuracy (%)', inplace=True)
  max_accuracy.index.rename('Learning Rate', inplace=True)
  print(max_accuracy[1:])

  # Save results to CSV
  test4.to_csv('test4.csv')


  plt.close()
  plt.figure(figsize=(12,10))
  epochs = np.arange(0,31)

  # Define different line styles and colors
  line_styles = ['-', '--', '-.', ':', '-']
  colors = ['red', 'green', 'blue', 'orange', 'purple']
  plt.figure(figsize=(12, 8))

  # Plot results
  for lr, color, style in zip(list(test4.columns[1:]), colors, line_styles):
      x = test4.index.tolist()[1:]
      y = test4[str(lr)][1:]
      plt.plot(x, y, marker='o', color=color, linestyle=style, label=f'{lr}', alpha=0.6)

  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Accuracy vs. Epoch for Different Learning Rates (Custom)')
  plt.legend()
  plt.grid()
  plt.autoscale()
  plt.tight_layout
  plt.savefig('test4 acc results.png')
  plt.show()



def run_nn(NInput, NHiden, NOutput, training_data, testing_data, learning_rate=3, num_epochs=30, minibatch_size=20):
    """ Initialize a NeuralNetwork with NInput input nodes, NHidden hidden nodes and NOutput output nodes.
    Then train the network on training_data, where training_data = X_train, D_train,
    and for each epoch of the training, tests the network on testing_data = X_test, D_test.
    
    The network's parameters are set using learning_rate, num_epochs, and minibatch_size, but defeault to 3, 30, 20.

    After the network is trained and tested like this, the accuracy is calculated for each epoch.
    The maximum accuracy is printed as well as the epoch at which that occurs.

    The data is saved to a CSV.

    Then, the test results are plotted and a graph is shown and saved as: f'Accuracies for Each Epoch (minibatch size = {minibatch_size}, learning rate = {learning_rate}).png' """
    
    print((training_data[0].shape[0]), 'training samples.')
    print((testing_data[0].shape[0]), 'testing samples.')

    print('Initializing layers')
    hidden_layer = Layer(NInput, NHidden)
    output_layer = Layer(NHidden, NOutput)

    print('Initializing Neural Network')
    nn = NeuralNetwork(hidden_layer, output_layer)
    learning_rate = 3
    num_epochs = 30
    minibatch_size = 20
    print(f'Training Network, num epochs = {num_epochs}, learning rate = {learning_rate}, minibatch size = {minibatch_size}')
    stats = nn.train(training_data, testing_data, learning_rate, num_epochs, minibatch_size)
    accuracies = np.array(stats)/(testing_data[1].shape[0]*testing_data[1].shape[1])*100
    print(f'Maximum Accuracy = {np.max(accuracies)}%, occurs at epoch {np.argmax(accuracies)}')     
    
    # SAVE DATA TO CSV
    test1_accuracies = np.array(stats)
    test1 = pd.Series(test1_accuracies)
    test1.index.rename('Epoch', inplace=True)
    test1.rename('Accuracy (%)', inplace=True)
    test1.to_csv("test1.csv")

    print('Plotting Test Results')
    plt.figure(figsize=(10,8))
    y = accuracies[1:]
    x = np.arange(1,num_epochs+1,dtype=int)
    plt.plot(x, y, marker='o',linestyle='dashed')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracies for Each Epoch (minibatch size = {minibatch_size}, learning rate = {learning_rate})')
    plt.grid()
    plt.savefig(f'Accuracies for Each Epoch (minibatch size = {minibatch_size}, learning rate = {learning_rate}).png')
    plt.show()
    print('Done')





# Check if the correct number of arguments are provided
if len(sys.argv) != 6 and len(sys.argv) != 7:

  # nn.py 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
  print("Usage: nn.py NInput NHidden NOutput training_data_filename testing_data_filename [any character to run tests]")

  sys.exit(1)
elif len(sys.argv) == 6:
  # Access the command-line arguments
  args = sys.argv

  # Extract the arguments
  NInput = int(args[1])
  NHidden = int(args[2])
  NOutput = int(args[3])
  train_filename = args[4]
  test_filename = args[5]
  train_data, test_data = get_train_test_data(train_filename, test_filename, NOutput)
  run_nn(NInput, NHidden,NOutput,train_data,test_data)

elif len(sys.argv) == 7:
  # Access the command-line arguments
  print('Running tests')
  args = sys.argv

  # Extract the arguments
  NInput = int(args[1])
  NHidden = int(args[2])
  NOutput = int(args[3])

  train_filename = args[4]
  test_filename = args[5]
  train_data, test_data = get_train_test_data(train_filename, test_filename, NOutput)
  print('---Test 1---')
  run_nn(NInput, NHidden,NOutput,train_data,test_data)
  print('Test 1 complete')
  print('---Test 2---')
  test_2(train_data, test_data)
  print('Test 2 complete')
  print('---Test 3---')
  test_3(train_data, test_data)
  print('Test 3 complete.')
  print('---Custom Test (test 4.2 in results section)---')
  test_4(train_data,test_data)
  print('Custom test complete. All tests now finished.')
