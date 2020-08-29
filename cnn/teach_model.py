import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers

images_data = []
expected_data = []


# Create convolutional neural network model.
# Output is 1 length arrays in single array
# https://www.tensorflow.org/tutorials/images/cnn
model = keras.Sequential()
model.add(layers.Conv2D(75, (3, 3), activation='relu', input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(125, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(125, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Imagedatagenerator for reading images from folders
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Read data from folders data/train/0 and data/train/1
train_gen = datagen.flow_from_directory(
    'data/train',
    shuffle=True,
    target_size=(250, 250),
    batch_size=10,
    class_mode='binary')

# Read data from folders data/validate/0 and data/validate/1
validation_gen = ImageDataGenerator(rescale=1.255).flow_from_directory(
    'data/validate',
    shuffle=True,
    target_size=(250, 250),
    batch_size=10,
    class_mode='binary')

# Save best model -callback
checkpoint = keras.callbacks.ModelCheckpoint("data/best", monitor='val_loss', save_best_only=True)

# Train
model.fit(train_gen, epochs=50, validation_data=validation_gen, callbacks=[checkpoint])

# Save last, but can be removed. Not used
model.save('data/model')
