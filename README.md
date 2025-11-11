### How to run the server
PS configure the database, the creation query is in /server/MAIZEDISEASE.sql, configure file Maize Disease Detection System\server\DB_integration.py as per your credentials.


1. Open the folder **maize detection system** in your IDE(VSCode as by creators)
2. Run the command **uvicorn api.main:app --reload**
3. in your web browser enter URL : http://127.0.0.1:8000 (you are now in the landing page)
4. drag and drop, open folder... your follow the UI on how to upload images to diagnose your maize plant
5. wait for the system to load your response
6. view your results



#  Maize Disease Detection System

An **AI-powered computer vision system** built with **FastAPI** and **deep learning** for detecting and describing maize (corn) leaf diseases.  
This project integrates **machine learning**, **RESTful API deployment**, and **OpenAI-assisted reporting** into a full pipeline — from dataset preprocessing to real-time prediction.

---

## Project Overview

The system allows users to upload an image of a maize leaf.  
The backend processes the image, predicts the disease using a **Convolutional Neural Network (CNN)** (ResNet50-based model), and generates a short description and treatment advice using the **OpenAI API**.

The results are then stored in a **SQL Server database** along with image metadata for further reference.

---

## Machine Learning Component

### Dataset
A publicly available maize leaf disease dataset was used (from Kaggle).  
It contains images of healthy and infected maize leaves with multiple disease categories, including:

- **Common Rust**
- **Gray Leaf Spot**
- **Northern Leaf Blight**
- **Healthy Leaves**

**Justification:**  
The dataset was chosen because of its high-quality labeled images, balanced class representation, and relevance to real-world agricultural issues.

---

### Model Development

#### Architecture:
A **transfer learning** approach was used with **ResNet50**, a deep CNN pre-trained on ImageNet.  
The final dense layers were fine-tuned for maize disease classification.

#### Key Steps:
1. **Image Preprocessing**
   - Resized all images to 224×224 pixels.
   - Normalized pixel values between 0 and 1.
   - Applied data augmentation (rotation, flipping, zooming) to reduce overfitting.

2. **Training**
   - Base model: `ResNet50(weights='imagenet', include_top=False)`
   - Added:
     - GlobalAveragePooling2D
     - Dense(128, activation='relu')
     - Dropout(0.5)
     - Dense(num_classes, activation='softmax')  
   - Optimizer: `Adam(lr=0.0001)`
   - Loss: `categorical_crossentropy`

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, and F1-score
   - Confusion Matrix
   - Classification Report
   - Loss and Accuracy Curves (visualized via Matplotlib)

4. **Challenges & Solutions**
   - **Overfitting** → Solved via data augmentation & dropout layers.
   - **Class imbalance** → Used weighted sampling and augmentation.
   - **Limited training data** → Leveraged transfer learning with ResNet50.

---

### Model Output Example
| Disease Class | Accuracy | Precision | Recall | F1-score |
|----------------|-----------|------------|----------|-----------|
| Common Rust | 0.94 | 0.93 | 0.95 | 0.94 |
| Gray Leaf Spot | 0.91 | 0.90 | 0.89 | 0.89 |
| Northern Blight | 0.93 | 0.92 | 0.91 | 0.91 |
| Healthy | 0.97 | 0.96 | 0.98 | 0.97 |

---

## FastAPI Deployment

### API Endpoints

| Method | Endpoint | Description |
|---------|-----------|-------------|
| `POST` | `/analyze/` | Upload an image for analysis and prediction |

### Example Response:
```json
{
  "image_name": "A1d2f3Z4x9KlmN.jpg",
  "predicted_disease": "Gray Leaf Spot",
  "openai_response": "Gray leaf spot is a fungal disease caused by *Cercospora zeae-maydis*..."
}
"# GROUP-12" 
