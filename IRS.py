import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load the image
img_path = 'Shark.jpg'  # Replace with your image path

try:
    print("Loading image...")
    img = image.load_img(img_path, target_size=(224, 224))
    print("Image loaded successfully!")

    # Convert the image to an array and preprocess it
    print("Preprocessing image...")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    print("Image preprocessed. Shape:", img_array.shape)

    # Predict
    print("Making predictions...")
    predictions = model.predict(img_array)

    print("Predictions obtained. Decoding...")
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    print("Decoded predictions:")
    for _, label, score in decoded_predictions:
        print(f"{label}: {score:.2f}")

except Exception as e:
    print("Error occurred:", e)

