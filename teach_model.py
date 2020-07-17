import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import glob
from read_images import image_to_list


images_data = []
expected_data = []

# Read folders
images_0 = glob.glob('data/0/*.jpg', recursive=True)
images_1 = glob.glob('data/1/*.jpg', recursive=True)

# read classified images to lists and append to single list
# Class 0
for image in images_0:
    images_data.append(image_to_list(image))
    expected_data.append([0.0])

# Class 1
for image in images_1:
    images_data.append(image_to_list(image))
    expected_data.append([1.0])

# Format ro numpy arrays
input_data = np.array(images_data)
expected_output = np.array(expected_data)

# Create simple neural network model.
# 20x20 = 400 => Input shape is 400 length arrays in single array
# Output is 1 length arrays in single array
model = keras.Sequential()
model.add(layers.Dense(400, activation="relu", input_shape=(400,)))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# compile and learn
model.compile(optimizer='sgd', loss='mse')
model.fit(input_data, expected_output, batch_size=100, epochs=1000)

# Save to folder
model.save('data/model')
