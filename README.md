🧬 Cancer Risk Prediction using Machine Learning
📌 Project Overview

This project focuses on building a Machine Learning based Cancer Risk Prediction System using feature selection techniques and predictive modeling. The objective is to improve prediction accuracy by selecting the most relevant medical features and comparing multiple machine learning algorithms.

The system includes:

Data preprocessing and feature engineering

Feature selection methods

Model training and evaluation

Deployment using a Flask API

🎯 Objectives

Apply feature selection to improve model performance.

Train and compare multiple machine learning models.

Evaluate models using standard performance metrics.

Deploy the best-performing model as an API.

🧠 Algorithms Used

The following machine learning algorithms were implemented:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

🔎 Feature Selection Techniques

The project uses multiple feature selection approaches:

Chi-Square Selection

Mutual Information Selection

Recursive Feature Elimination (RFE)

These techniques help reduce dimensionality and improve predictive performance.

🏗️ Project Structure
CancerRiskPrediction/
│
├── src/
│   ├── dataloader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   └── train_pipeline.py
│
├── app.py
├── main.py
├── config.py
├── best_model.pkl
└── requirements.txt
⚙️ Installation
Clone the Repository
git clone https://github.com/Tejeswar-Reddy1234/cancer-risk-prediction-ml.git
cd cancer-risk-prediction-ml
Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Train the Model
python main.py

This will:

Load dataset

Perform preprocessing

Apply feature selection

Train models

Save the best model as best_model.pkl

Run the Flask API
python app.py

Server will start at:

http://127.0.0.1:5000
🌐 API Usage
Home Route
GET /

Response:

Cancer Prediction API Running
Prediction Route
POST /predict

Example Request:

{
  "features": [10.1, 15.2, 80.3, 500.2]
}

Example Response:

{
  "prediction": 1
}
📊 Evaluation Metrics

Models are evaluated using:

Accuracy

Precision

Recall

F1 Score

The best performing model is automatically saved.

🧪 Dataset

This project uses a structured medical dataset for cancer prediction containing numerical features representing tumor characteristics.

🚀 Technologies Used

Python

Scikit-learn

NumPy

Flask

Joblib

👨‍💻 Authors

N. Vaibhav Kumar

Kohith Pappala

D. Surendra Reddy

Sairam Reddy

Tejaswar Reddy

🏫 Institution

Woxsen University
School of Technology
B.Tech Artificial Intelligence and Machine Learning

📜 License

This project is developed for academic purposes only.
