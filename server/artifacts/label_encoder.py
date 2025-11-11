import os
import joblib

# Path to your dataset directory
data_dir = r"C:\Users\USER\Documents\AI\computervision\data\maize"

# Get class names in alphabetical order
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Create mapping
class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Save the mapping to a .joblib file
os.makedirs("artifacts", exist_ok=True)
joblib.dump(class_indices, r"C:\Users\USER\Documents\AI\computervision\artifacts\label_encoder.joblib")

print("✅ Label encoder saved at artifacts/label_encoder.joblib")
print("Class indices:", class_indices)
