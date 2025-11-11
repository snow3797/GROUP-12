# src/utils.py
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib


def load_images_from_folder(folder, target_size=(224,224), max_per_class=None):
    X = []
    y = []
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))])
    for cls in classes:
        cls_folder = os.path.join(folder, cls)
        files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('jpg','jpeg','png'))]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                im = Image.open(f).convert('RGB')
                im = im.resize(target_size)
                arr = np.array(im)
                X.append(arr)
                y.append(cls)
            except Exception as e:
                print(f"Skipped {f}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X,y


def save_label_encoder(y, path):
    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, path)
    return le


def load_label_encoder(path):
    return joblib.load(path)