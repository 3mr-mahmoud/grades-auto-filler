import os
from sklearn import svm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

# Resolve dataset path
path = "../../../Dataset/Training Set/digits_dataset"

target_img_size = (32, 32)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

def load_dataset(path_to_dataset):
    if not os.path.exists(path_to_dataset):
        raise FileNotFoundError(f"Dataset path does not exist: {path_to_dataset}")
    
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)

    for i, fn in enumerate(img_filenames):
        if fn.split('.')[-1] != 'jpg':
            continue

        label = fn.split('.')[0]
        labels.append(label)

        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path)
        features.append(extract_hog_features(img))
    return features, labels

def train(path_to_dataset=path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_dataset = os.path.join(script_dir, path_to_dataset)
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(path_to_dataset)
    print('Finished loading dataset.')
    
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)
    
    model = svm.LinearSVC(random_state=random_seed)
    model.fit(train_features, train_labels)

    # Evaluate the model
    predictions = model.predict(test_features)
    print("Accuracy:", accuracy_score(test_labels, predictions))

    return model