# Car Evaluation with Decision Tree Learning From Scratch
The aim of this project was to implement a learning decision tree from scratch which could correctly evaluate cars based on
the 6 features in the car.csv dataset. The car.csv dataset includes 1728 cars with features: buying
price, cost of maintenance, number of doors, capacity in terms of persons to carry, the relative size of
luggage boot and the estimated safety of each car. The final column of the dataset evaluates the car
into 1 of 4 classes: unacceptable, acceptable, good or very good.

Everything to do with the model including the train/test split function was implemented from scratch.

The program requires matplotlib, seaborn, numpy and pandas.

# Model Performance
![precision_f1_and_recall.png](model_performance_evaluation_outputs/precision_f1_and_recall.png)
![confusion_matrix.png](model_performance_evaluation_outputs/confusion_matrix.png)
![learning_curve.png](model_performance_evaluation_outputs/learning_curve.png)
This implementation ID3 learning decision tree performed very well with the dataset used. The
tree was implemented as a binary tree. Since the model consistently performs at a high level of
accuracy (mid to high 90s), adding more branches would unnecessarily increase the complexity of the model. The
test set did not have an even distribution of classes, with 72.2% of all
test feature vectors belonging to the unacc class. This means the model evaluations may be subject
to bias, and may not generalize well to a larger test set which has a more even distribution.
Additionally, the distribution of the training data was never verified, and it could have similar
problems to the test set. Therefore, to improve the model, the distribution of the training data
should be explored, and the model should be tested on more examples for the other classes,
particularly the good and vgood classes. Overall, for the needs of this dataset, the model
performance was exemplary based upon the precision, F1 and recall scores and the confusion matrix.

# Instructions
The program will print the train test split and the accuracy of the model by calling the `main()` function with the filename of the input data csv, the training and testing data
sizes, and whether the user wants performance evaluation plots to be shown upon completion or not. 



# Building the Tree
The binary decision tree is built recursively by the ```id3_build_tree()``` function with inputs `X` and `y` which
are arrays of the input and outputs of the training data. `id3_build_tree()` returns the root node of the decision tree. Then, to make a decision
for feature vector `x` based off the decision tree, `make_decision()` can be called with the root node
and x passed through. This will return the decision for x.