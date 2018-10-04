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

# one hot encode
number_of_classes = 10
Y_train = keras.utils.to_categorical(Y_train, number_of_classes)
Y_val = keras.utils.to_categorical(Y_val, number_of_classes)

# Reshape to [samples][pixel_rows][pixel_cols][channels]
X_train = np.array(X_train).reshape(len(X_train), 28, 28,1)
X_val = np.array(X_val).reshape(len(X_val), 28, 28,1)
X_test = np.array(X_test).reshape(len(X_test), 28, 28,1)


# look at shapes
X_train.shape
Y_train.shape

X_val.shape
Y_val.shape



# Scale the pixel data
X_train.max()
X_train, X_val,X_test = X_train/255 , X_val/255,X_test/255

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
    tf.keras.layers.Dense(10, activation = tf.nn.softmax),
]
)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Fit using generated data
batch_size = 100

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=20,
                        validation_data=(X_val, Y_val))


plt.figure(figsize = (10,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')



# CNN


model_CNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(28, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=[28,28,1]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
]
)


model_CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Fit using generated data
batch_size = 100

history = model_CNN.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=20,
                        validation_data=(X_val, Y_val))

plt.figure(figsize = (10,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# Use conv NN

Y_test = model.predict(X_test)

Y_test.shape

[np.where(r==1)[0] for r in Y_test]

index

pd.Series(index)

pd.to_numeric(pd.Series(Y_test),downcast='integer')

output = pd.concat([pd.Series(range(X_test.shape[0]),pd.to_numeric(pd.Series(Y_test),downcast='integer')], axis = 1)

output.head()

output.columns = ['ImageId','Label']

output.to_csv('output.csv',index=False)
