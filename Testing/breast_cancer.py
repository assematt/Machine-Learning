# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, model_selection

# Load the breast cancer dataset
cancerData = datasets.load_breast_cancer()

# Create a training and testing set of labels/features where 10% of the data is used for training
X_train, X_test, y_train, y_test = _train, X_test, y_train, y_test = model_selection.train_test_split(cancerData.data, cancerData.target, train_size=0.1)

# We use a Support Vector Machine with a linear kernel
classifier = svm.SVC(kernel="linear")

# Train the classifier
classifier.fit(X_train, y_train)

# Tries to classifies the remaing 90% of the breast cancer data set
y_pred = classifier.predict(X_test)

# Print some stats
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
print ("Accuracy: %.2f" % metrics.accuracy_score(y_test, y_pred))