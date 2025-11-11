import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# 1. Configuration
# -------------------------------
MODEL_PATH = r"C:\Users\USER\Documents\AI\computervision\src\models\cnn_model.keras"
LABEL_ENCODER_PATH = r"C:\Users\USER\Documents\AI\computervision\artifacts\label_encoder.joblib"
DATA_DIR = r"C:\Users\USER\Documents\AI\computervision\data\maize"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SAVE_DIR = "artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# 2. Load model and label encoder
# -------------------------------
model = load_model(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
inv_map = {v: k for k, v in label_encoder.items()}
class_names = list(label_encoder.keys())
num_classes = len(class_names)

# -------------------------------
# 3. Prepare validation data
# -------------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

# -------------------------------
# 4. Make predictions
# -------------------------------
preds_proba = model.predict(val_gen)
y_pred = np.argmax(preds_proba, axis=1)
y_true = val_gen.classes

# -------------------------------
# 5. Compute overall metrics
# -------------------------------
metrics_dict = {
    "Accuracy": [accuracy_score(y_true, y_pred)],
    "Precision": [precision_score(y_true, y_pred, average='weighted')],
    "Recall": [recall_score(y_true, y_pred, average='weighted')],
    "F1 Score": [f1_score(y_true, y_pred, average='weighted')]
}

df_metrics = pd.DataFrame(metrics_dict)

# -------------------------------
# 6. Save overall metrics as image
# -------------------------------
fig, ax = plt.subplots(figsize=(6,2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_metrics.values,
                 colLabels=df_metrics.columns,
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.title("Model Evaluation Metrics", fontsize=14)
plt.savefig(f"{SAVE_DIR}/model_metrics.png", bbox_inches='tight')
plt.show()

# -------------------------------
# 7. Per-class classification report
# -------------------------------
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose().round(3)

# -------------------------------
# 8. Save per-class metrics as image
# -------------------------------
fig, ax = plt.subplots(figsize=(8, len(df_report)*0.5 + 1))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_report.values,
                 colLabels=df_report.columns,
                 rowLabels=df_report.index,
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Per-Class Metrics", fontsize=14)
plt.savefig(f"{SAVE_DIR}/per_class_metrics.png", bbox_inches='tight')
plt.show()
