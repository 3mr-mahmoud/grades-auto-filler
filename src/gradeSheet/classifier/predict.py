from .train import extract_hog_features
import numpy as np

def predictGeneral(cell, model):
    hogFeatures = extract_hog_features(cell)
    # Reshape the feature to be 2D (1 sample, n features)
    hogFeatures = hogFeatures.reshape(1, -1)
    prediction = model.predict(hogFeatures)
    print("Prediction: ", prediction[0])
    return prediction
    
