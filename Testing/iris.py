# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, model_selection

# Load the iris dataset
irisData = datasets.load_iris()

# Create a training and testing set of labels/features where 60% of the data is used for training
X_train, X_test, y_train, y_test = _train, X_test, y_train, y_test = model_selection.train_test_split(irisData.data, irisData.target, train_size=0.6)

# We use a Support Vector Machine with default values 
classifier = svm.SVC()

# Train the classifier
classifier.fit(X_train, y_train)

# Tries to classifies the remaing 40% of the iris data set
y_pred = classifier.predict(X_test)

# Print some stats
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
print ("Accuracy: %.2f" % metrics.accuracy_score(y_test, y_pred))