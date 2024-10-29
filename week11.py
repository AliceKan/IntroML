import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#print(x_train.shape)

#plt.imshow(x_train[2])
#plt.show()

x_train, x_test = x_train /255.0, x_test/255.0

model = models.Sequential([
	layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
	layers.MaxPooling2D((2,2)),
	layers.Conv2D(128, (4,4), activation='relu'),
	layers.AveragePooling2D((2,2)),
	layers.Conv2D(64, (3,3), strides=(2,2), activation='relu'),
	layers.MaxPooling2D((2,2)),
	layers.Flatten(),
	layers.Dense(64, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10, activation='softmax')])

#model.summary()

model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=128)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()



