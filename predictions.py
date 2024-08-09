import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Function to load and preprocess the image
def preprocess_image(img_path):
    # Load the image with the target size
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Normalize the image
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Function to predict the class of the image
def predict_image(img_path, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class, predictions

# Path to the image you want to predict
img_path = '/content/drive/MyDrive/trainset/false/20230102-1.png'  # Replace with your image path

# Make a prediction
predicted_class, predictions = predict_image(img_path, model)

# Assuming you have a list of class names corresponding to the integer labels
class_names = ['false', 'true']  # Replace with your actual class names

# Print the result
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Prediction probabilities: {predictions}")
