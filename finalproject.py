from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

# load Fashion-MNIST dataset
fashion_mnist = datasets.fetch_openml('Fashion-MNIST', version=1)

# the keys in the dataset dictionary
#print("Keys:", fashion_mnist.keys())

# into arrays
X = np.array(fashion_mnist.data)
Y = np.array(fashion_mnist.target)

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/7, random_state=0)

# check shapes
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# try training
start = time.time()
classifier = svm.SVC().fit(X_train, Y_train)
acc = classifier.score(X_test, Y_test)
print('Accuracy = ', acc)

# for the confusion matrix and other metrics
predicted_labels = classifier.predict(X_test)
end = time.time()
print(end - start)
conf_matrix = confusion_matrix(Y_test, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)
accuracy = accuracy_score(Y_test, predicted_labels)
precision = precision_score(Y_test, predicted_labels, average='weighted')
recall = recall_score(Y_test, predicted_labels, average='weighted')
f1 = f1_score(Y_test, predicted_labels, average='weighted')
report = classification_report(Y_test, predicted_labels)

# print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", report)


