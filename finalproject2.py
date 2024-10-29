import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# load data and see the shapes
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# scale pixel value
x_train, x_test = x_train /255.0, x_test/255.0

# build the CNN model
model = models.Sequential([
	layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),
	layers.MaxPooling2D((2,2)),
	layers.Conv2D(64, (3,3), activation='relu'),
	layers.AveragePooling2D((2,2)),
	layers.Flatten(),
	layers.Dense(64, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10, activation='softmax')])

# see the summary
#model.summary()

# compilation
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

# training
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=128)

# for confusion matrix and other metrics
y_pred = model.predict(x_test)
# max prob
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", conf_matrix)
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average='weighted')
recall = recall_score(y_test, y_pred_labels, average='weighted')
f1 = f1_score(y_test, y_pred_labels, average='weighted')
report = classification_report(y_test, y_pred_labels)

# print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", report)

# the accuracies vs epochs plotting
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

