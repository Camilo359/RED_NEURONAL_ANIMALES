import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as incv3_preproc
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preproc
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf

# Path to the saved .h5 model
MODEL_PATH = "C:/Users/JORFAR.JORFAR/Documents/red neuronal convolucional/best_model.keras"

# Load the saved .h5 model
model = load_model(MODEL_PATH)

# Load the dataframe containing dog breeds (replace with actual path to your CSV file)
import pandas as pd
df = pd.read_csv('C:/Users/JORFAR.JORFAR/Documents/red neuronal convolucional/labels.csv')

# Dog breeds (ensure this matches the list from your training pipeline)
dog_breeds = sorted(df['breed'].unique())

# Pretrained model input size
IMG_SIZE = (299, 299, 3)

# Function to extract features using pretrained models
def extract_features(model_class, preproc_func, input_size, data):
    """
    Extract features using a pretrained model.
    """
    # Create a Lambda layer for preprocessing
    base_input = tf.keras.layers.Input(shape=input_size)
    x = tf.keras.layers.Lambda(preproc_func)(base_input)
    
    # Instantiate the pretrained base model
    base_model = model_class(weights='imagenet', include_top=False, input_shape=input_size)(x)
    
    # Apply Global Average Pooling
    out = GlobalAveragePooling2D()(base_model)
    
    # Create feature extractor
    feature_extractor = tf.keras.models.Model(base_input, out)
    
    # Extract features
    feats = feature_extractor.predict(data, batch_size=1, verbose=1)
    return feats

# Function to preprocess and predict a single image
def predict_single_image(img_path, model, dog_breeds):
    """
    Predict the breed of a single image using the .h5 model.
    """
    # 1. Load & resize image
    img = load_img(img_path, target_size=IMG_SIZE[:2])
    img_array = np.array(img)
    
    # 2. Expand dims to make it (1, height, width, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Extract features from pretrained models
    feats_incv3 = extract_features(InceptionV3, incv3_preproc, IMG_SIZE, img_array)
    feats_xcep  = extract_features(Xception, xception_preproc, IMG_SIZE, img_array)

    # 4. Concatenate features
    final_feats = np.concatenate([feats_incv3, feats_xcep], axis=-1)
    
    # 5. Predict breed
    preds = model.predict(final_feats)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_breed = dog_breeds[pred_idx]
    print(f"Predicted breed: {pred_breed}")

# Test image path
test_image_path = "C:/Users/JORFAR.JORFAR/Documents/red neuronal convolucional/test/fedd9cc1cf0075443e7dfc349d24d412.jpg"

# Predict breed for the test image
predict_single_image(test_image_path, model, dog_breeds)