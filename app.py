from flask import Flask,render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os

UPLOAD_FOLDER = os.path.join('static','uploads')

labels_csv = pd.read_csv("data/labels.csv")
labels = labels_csv["breed"]
unique_breeds = np.unique(labels)

def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a prediction label
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

# Define image size
IMG_SIZE = 224
# Create a function for preprocessing images
def process_image(image_path):
  """
  Takes an image and turns the image into a Tensor
  """
  # Read in an image file
  image = tf.io.read_file(image_path)

  # Turn the jpg image into a numerical tensor with three color channels (RGB)
  image = tf.image.decode_jpeg(image, channels = 3)

  # Convert the colour channel values from 0-255 values to 0-1 values (Image Representation Using Floating Points Needs 0-1 Values)
  image = tf.image.convert_image_dtype(image, tf.float32) # Normalization !

  # Resize our image to our desired size (224,224)
  image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])

  return image

# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path and the associated label
  Processes image and returns tuple of (image, label) 
  """
  image = process_image(image_path)
  return image, label

# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn our data into batches
def create_data_batches(X, batch_size = BATCH_SIZE):

  print("Creating test data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # Only filepaths, no labels (Makes dataset from slices of given tensor)
  data_batch = data.map(process_image).batch(BATCH_SIZE) # Creates batches of given batch size
  return data_batch

  
# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path
  """
  print(f"Loading Saved Model From: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {"KerasLayer": hub.KerasLayer})
  return model

model = load_model("./models/model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
  return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def prediction():
    file = request.files['file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    image_path = [save_path]
    image = create_data_batches(image_path)
    prediction = model.predict(image)
    pred_label = [get_pred_label(prediction)]
    return render_template('results.html', breed=pred_label[0].replace("_"," ").title())
  
if __name__ == "__main__":
    app.run()