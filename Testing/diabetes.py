# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, model_selection

# Load the diabetes dataset
diabetesData = datasets.load_diabetes()

# Create a training and testing set of labels/features where 10% of the data is used for training
X_train, X_test, y_train, y_test = _train, X_test, y_train, y_test = model_selection.train_test_split(diabetesData.data, diabetesData.target, train_size=0.95, shuffle=False)

# We use a Support Vector Machine with a linear kernel
classifier = linear_model.LinearRegression()

# Train the classifier
classifier.fit(X_train, y_train)

# Tries to classifies the remaing 90% of the breast cancer data set
y_pred = classifier.predict(X_test)

# Print some stats
print ("Accuracy: %.2f" % classifier.score(X_test, y_test))