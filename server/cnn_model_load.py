import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_leaf_class(image_path,
                       model_path=r'server\src\models\cnn_model.keras',
                       label_encoder_path=r'server\artifacts\label_encoder.joblib'):
    """
    Predicts the class of a single leaf image using a trained CNN model.
    
    Parameters:
        image_path (str): Path to the image file to classify.
        model_path (str): Path to the trained model file (.keras or .h5).
        label_encoder_path (str): Path to the saved label encoder (.joblib).

    Returns:
        str: Predicted class label.
    """

    # Load trained model and label encoder
    model = load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)
    inv_map = {v: k for k, v in label_encoder.items()}

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Make prediction
    pred = model.predict(x)
    predicted_class = inv_map[np.argmax(pred)]

    print(f"Predicted class for '{image_path}': {predicted_class}")
    return predicted_class
