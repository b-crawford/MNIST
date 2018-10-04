import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# import data
X_test = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

X_test.shape
train_df.shape

# split the training data into training and vaidation
train_df,val_df = train_test_split(train_df)

train_df.shape
val_df.shape

# seperate out the X and Y data frames
X_train = train_df.drop(['label'], axis = 1)
Y_train = train_df['label']

X_val = val_df.drop(['label'], axis = 1)
Y_val = val_df['label']

# Reshape to [samples][pixel_rows][pixel_cols][channels]
X_train = np.array(X_train).reshape(len(X_train), 28, 28,1)
X_val = np.array(X_val).reshape(len(X_val), 28, 28,1)


# Scale the pixel data
X_train.max()
X_train, X_val = X_train/255 , X_val/255

X_train.max()



# Random rotations
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45)
datagen.fit(X_train)

# Neural network:

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = [28,28,1]),
    tf.keras.layers.Dense(512, activation = tf.nn.relu),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(56, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(56, activation = tf.nn.softmax),
]
)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Fit using generated data
batch_size = 100

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=20,
                        validation_data=(X_val, Y_val))


plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
