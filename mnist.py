import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

batch_size = 128
num_classes = 10
epochs = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)#轉成二維陣列
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')#原本是<class 'numpy.uint8'>
x_test = x_test.astype('float32')
# x_test/=255
# x_train/=255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([keras.Input(shape=(784,))])
model.add(Dense(512, activation="relu", ))

model.add(Dense(512, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy', 'precision', 'recall', 'f1_score'])
train_history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_split = 0.2)
y_pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose = 0)
print('test Acc: ', round(score[1], 2), " %")#損失值用於衡量模型預測的品質，越小越好
confusion = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred,axis=1))

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 
                    'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
print('\n')
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred,axis=1), 
        target_names=target_names))
print('\n')
# print(confusion)   
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.xticks([i for i in range(0, len(train_history.history['accuracy']))])
plt.title('Train History')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()