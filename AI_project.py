import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = MobileNetV2(weights='imagenet')

# Load the image
img_path = 'img.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 input size

# Convert the image to an array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=5)[0]

for _, label, score in decoded_predictions:
    print(f"{label}: {score:.2f}")