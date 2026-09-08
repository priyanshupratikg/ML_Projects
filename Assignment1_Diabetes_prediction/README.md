
# 🩺 DiabDetect – Diabetes Risk Prediction System

### Machine Learning & Multilayer Perceptron Based Diabetes Prediction

**Developed by Team Vajra**

DiabDetect is a machine-learning-based diabetes risk prediction system developed as an academic project. The system analyzes clinical measurements provided by a user and predicts whether the individual is likely to belong to the diabetic or non-diabetic class.

The project demonstrates the complete machine learning lifecycle:

- Dataset understanding
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Missing/suspicious value handling
- Outlier analysis
- Feature scaling
- Machine learning model development
- Multilayer Perceptron (MLP) development
- Model comparison
- Hyperparameter tuning
- Final model selection
- Model persistence
- Streamlit deployment

---

# 🌐 Live Demo

## 🚀 Access DiabDetect Online

The application is deployed using Streamlit Community Cloud and can be accessed directly through a web browser.

**Live Application:**

https://diabdetect-vajra.streamlit.app/

No local installation or VS Code setup is required to use the deployed application.

---

# 👥 Team Vajra

| Team Member | Roll Number |
|---|---|
| **Priyanshu Partik** | 2341016436 |
| **Tushar Mallick** | 2341013092 |
| **Aditya Prasad Mishra** | 2341013147 |
| **Sarbajeet Sahoo** | 2341019148 |

---

# 📌 Project Overview

Diabetes is a chronic medical condition in which the body is unable to properly regulate blood glucose levels. Early identification of diabetes risk can help support timely medical attention and lifestyle management.

The objective of DiabDetect is to build a predictive classification system using patient clinical measurements.

The project uses the **Pima Indians Diabetes Dataset**, a commonly used dataset for demonstrating binary classification techniques.

The system predicts one of two classes:

- **0 → Non-Diabetic**
- **1 → Diabetic**

The final prediction is generated using a tuned Random Forest classification model.

---

# 🎯 Objectives

The main objectives of this project are:

1. Understand and analyze the diabetes dataset.
2. Identify missing, suspicious, and abnormal values.
3. Perform exploratory data analysis.
4. Handle missing values appropriately.
5. Analyze and handle outliers.
6. Prepare the data for machine learning.
7. Apply feature scaling.
8. Train multiple machine learning algorithms.
9. Build and evaluate a Multilayer Perceptron (MLP).
10. Compare different models using multiple evaluation metrics.
11. Tune the selected model.
12. Save the trained model and preprocessing objects.
13. Develop a user-friendly prediction interface.
14. Deploy the application using Streamlit Community Cloud.

---

# 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains:

- **768 observations**
- **8 input features**
- **1 target variable**

## Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | 2-Hour serum insulin |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age of the patient |
| `Outcome` | Target variable |

### Target Variable

`Outcome` is a binary classification variable:

```text
0 = Non-Diabetic
1 = Diabetic
````

---

# 🔍 Exploratory Data Analysis

Before training the models, the dataset was extensively analyzed.

The analysis included:

* Dataset dimensions
* Data types
* Statistical summary
* Target distribution
* Feature distributions
* Correlation analysis
* Missing/suspicious value analysis
* Outlier detection

EDA was performed to understand the characteristics of the dataset and identify issues that could negatively affect model performance.

---

# 🧹 Data Preprocessing

Data preprocessing was an important part of the project.

## 1. Suspicious Zero Values

Several clinical measurements cannot realistically have a value of zero.

The following features were therefore treated as having suspicious zero values:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

A zero in these columns was treated as a missing value rather than a valid measurement.

`Pregnancies = 0`, however, is a valid value and was retained.

---

## 2. Missing Value Handling

After identifying suspicious zero values, they were converted into missing values.

Median imputation was then applied.

### Why Median Imputation?

Median imputation was selected because it:

* Preserves the number of observations.
* Is simple and robust.
* Is less sensitive to extreme values than mean imputation.
* Works well when clinical variables contain outliers.

The trained imputer is saved as:

```text
models/median_imputer.pkl
```

---

# 📈 Outlier Analysis

Outliers were investigated using statistical and visual analysis.

Particular attention was given to clinical features such as:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* Age

Outliers were analyzed carefully rather than blindly removing them because extreme clinical values may represent genuine patient conditions.

---

# 📏 Feature Scaling

After missing-value treatment, feature scaling was applied using:

```text
StandardScaler
```

Standardization transforms features so that they have approximately:

```text
Mean = 0
Standard Deviation = 1
```

This is especially useful for models such as:

* Logistic Regression
* SVM
* KNN
* MLP

The trained scaler is saved as:

```text
models/standard_scaler.pkl
```

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and compared.

The project evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Support Vector Machine (SVM)
5. Random Forest
6. Gradient Boosting

A Multilayer Perceptron (MLP) was also developed as the neural-network-based model.

---

# 🧠 Multilayer Perceptron

A Multilayer Perceptron is a type of artificial neural network used for classification and regression tasks.

The MLP consists conceptually of:

```text
Input Layer
     ↓
Hidden Layer(s)
     ↓
Output Layer
```

The input layer receives the clinical features.

The hidden layers learn relationships between the features.

The output layer produces the final binary classification.

The MLP was evaluated independently and compared with traditional machine learning models.

---

# 🏆 Model Comparison

The main machine learning models produced the following evaluation results:

| Model               |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: |
| Gradient Boosting   |     88.8% |     78.4% |     93.0% |     85.1% |     96.7% |
| **Random Forest**   | **90.2%** | **80.4%** | **95.3%** | **87.2%** | **95.9%** |
| SVM                 |     87.8% |     78.2% |     88.4% |     83.5% |     94.2% |
| Logistic Regression |     82.9% |     82.4% |     65.1% |     72.7% |     91.0% |
| KNN                 |     84.6% |     78.6% |     76.7% |     77.6% |     90.2% |
| Decision Tree       |     81.3% |     69.2% |     83.7% |     77.8% |     81.9% |

Based on the evaluation results, **Random Forest** was selected as the final model.

---

# 🌲 Final Model – Random Forest

The final deployed prediction system uses a tuned:

```text
Random Forest Classifier
```

The final model uses:

```text
300 estimators
```

Random Forest was selected because it provided strong overall performance across the evaluation metrics, particularly accuracy, recall, and F1-score.

The trained model is stored as:

```text
models/final_random_forest.pkl
```

---

# 📦 Saved Model Artifacts

The project stores the trained model and preprocessing components so that the application does not need to retrain the model every time it starts.

```text
models/
│
├── final_random_forest.pkl
├── median_imputer.pkl
└── standard_scaler.pkl
```

### `final_random_forest.pkl`

Contains the trained final Random Forest classifier.

### `median_imputer.pkl`

Contains the fitted median imputation object used during preprocessing.

### `standard_scaler.pkl`

Contains the fitted StandardScaler used to standardize the input features.

---

# 🖥️ Streamlit Application

The user interface was developed using **Streamlit**.

The application provides a simple clinical assessment interface where the user can enter patient information and clinical measurements.

The application includes:

* DiabDetect branding
* Patient information section
* Gender selection
* Clinical measurements
* Prediction button
* Predicted diabetes class
* Estimated model probability
* Educational-use disclaimer

---

# 🔄 Prediction Pipeline

When a user enters clinical information, the application follows the same preprocessing pipeline used during model development.

```text
User Input
    ↓
Clinical Measurements
    ↓
Missing/Suspicious Value Handling
    ↓
Median Imputation
    ↓
Standard Scaling
    ↓
Trained Random Forest Model
    ↓
Prediction
    ↓
Probability Estimate
    ↓
Result Display
```

This ensures that the data supplied to the trained model is processed consistently with the training data.

---

# 📁 Project Structure

```text
ML_Projects/
│
├── Assignment1_Diabetes_prediction/
│   │
│   ├── app/
│   │   └── app.py
│   │
│   ├── models/
│   │   ├── final_random_forest.pkl
│   │   ├── median_imputer.pkl
│   │   └── standard_scaler.pkl
│   │
│   ├── notebooks/
│   │   └── diabetes_prediction.ipynb
│   │
│   └── ...
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

## Machine Learning

* Logistic Regression
* KNN
* Decision Tree
* SVM
* Random Forest
* Gradient Boosting
* Multilayer Perceptron

## Deployment

* Streamlit Community Cloud
* GitHub

---

# 📋 Requirements

The project dependencies are specified in:

```text
requirements.txt
```

Main dependencies include:

```text
streamlit==1.63.0
pandas
numpy
scikit-learn==1.6.1
joblib==1.5.3
```

---

# 💻 Running the Project Locally

## Step 1 – Clone the Repository

Clone the GitHub repository to your local machine.

## Step 2 – Navigate to the Project

Open a terminal in the project directory.

```bash
cd ML_Projects/Assignment1_Diabetes_prediction
```

## Step 3 – Install Dependencies

```bash
pip install -r ../requirements.txt
```

If the requirements file is located in the current directory, use:

```bash
pip install -r requirements.txt
```

## Step 4 – Run Streamlit

```bash
streamlit run app/app.py
```

The application will open in a browser.

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployment architecture is:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Python Environment
       ↓
Streamlit Application
       ↓
Public Web URL
```

The application is hosted independently of the development laptop.

Therefore, users do not need:

* VS Code
* Local Python environment
* Local Streamlit server
* ngrok
* The developer's laptop

to access the deployed application.

---

# 🌐 Live Application

### DiabDetect – Vajra

[https://diabdetect-vajra.streamlit.app/](https://diabdetect-vajra.streamlit.app/)

The application can be accessed from:

* Desktop
* Laptop
* Tablet
* Mobile browser

provided there is an internet connection.

---

# 📊 Evaluation Metrics

The models were evaluated using multiple classification metrics.

## Accuracy

Measures the overall percentage of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

## Precision

Measures how many of the observations predicted as positive were actually positive.

```text
Precision =
TP / (TP + FP)
```

## Recall

Measures how many of the actual positive cases were correctly identified.

```text
Recall =
TP / (TP + FN)
```

## F1-Score

The harmonic mean of precision and recall.

```text
F1 =
2 × Precision × Recall / (Precision + Recall)
```

## ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

---

# 🧪 MLP Evaluation

The developed MLP produced the following evaluation results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 88.62% |
| Precision | 82.22% |
| Recall    | 86.05% |
| F1-Score  | 84.09% |

The MLP confusion matrix was:

|                 | Predicted Negative | Predicted Positive |
| --------------- | -----------------: | -----------------: |
| Actual Negative |                 72 |                  8 |
| Actual Positive |                  6 |                 37 |

---

# 🔐 Model Persistence

Instead of retraining the model whenever the Streamlit application starts, the trained model is saved using Joblib.

This provides several advantages:

* Faster application startup
* Consistent predictions
* No need for model retraining during inference
* Easy deployment
* Separation between training and prediction

---

# 🧑‍💻 Application Workflow

The complete project workflow can be summarized as:

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Suspicious Value Detection
   ↓
Missing Value Treatment
   ↓
Outlier Analysis
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Multiple ML Models
   ↓
MLP Model
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Random Forest
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
Streamlit Cloud Deployment
```

---

# ⚠️ Important Disclaimer

DiabDetect is an **academic machine learning project** intended for educational and demonstration purposes.

The prediction generated by the application is an estimated machine-learning output and **must not be considered a medical diagnosis**.

The application should not be used as a substitute for professional medical advice, clinical examination, laboratory testing, or consultation with a qualified healthcare professional.

---

# 🎓 Academic Purpose

This project was developed as part of the **Machine Learning / Python Web Programming (MLPWP) Lab Assignment**.

The project demonstrates practical implementation of:

* Data preprocessing
* Exploratory data analysis
* Supervised machine learning
* Classification
* Neural networks
* Model evaluation
* Hyperparameter tuning
* Model persistence
* Web application development
* Cloud deployment

---

# ⭐ Key Highlights

### 🔹 Complete ML Pipeline

The project covers the complete journey from raw dataset to deployed application.

### 🔹 Multiple Models

Several traditional machine learning algorithms were compared rather than relying on a single model.

### 🔹 Neural Network

A Multilayer Perceptron was implemented and evaluated.

### 🔹 Robust Preprocessing

Suspicious zero values were identified and handled through median imputation.

### 🔹 Model Comparison

Models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

### 🔹 Final Model

A tuned Random Forest classifier was selected for the deployed application.

### 🔹 Cloud Deployment

The final application is publicly accessible through Streamlit Community Cloud.

---

# 🚀 Try DiabDetect

## Live Application

[https://diabdetect-vajra.streamlit.app/](https://diabdetect-vajra.streamlit.app/)

---

# 📚 Future Improvements

Possible future improvements include:

* Testing additional datasets
* Larger and more diverse clinical datasets
* Additional feature engineering
* More extensive hyperparameter optimization
* Cross-validation-based model comparison
* Explainable AI techniques such as SHAP
* Improved calibration of predicted probabilities
* Database integration
* User authentication
* Prediction history
* More comprehensive clinical decision-support features

---

# 📌 Conclusion

DiabDetect demonstrates how machine learning can be used to develop a complete diabetes-risk classification application.

The project combines data preprocessing, exploratory analysis, traditional machine learning algorithms, neural-network modeling, model comparison, model persistence, and cloud deployment into a single end-to-end system.

The final tuned Random Forest model provides the prediction engine for the deployed Streamlit application.

### 🌐 Live Demo

[https://diabdetect-vajra.streamlit.app/](https://diabdetect-vajra.streamlit.app/)

---

## 👥 Team Vajra

**Priyanshu Partik**
**Tushar Mallick**
**Aditya Prasad Mishra**
**Sarbajeet Sahoo**

---

### 🩺 DiabDetect

**Intelligent Diabetes Risk Prediction**
**By Vajra**

```

### One important thing

For GitHub, I'd recommend putting the **live app link right at the very top** of the README as a prominent **"🚀 Live Demo"** section. That way, your professor can open the repository and immediately click through to the working application.

Your actual live link is:

:contentReference[oaicite:2]{index=2}

And because your app is now hosted on Streamlit Cloud, **you can close VS Code and even shut down your laptop—the deployed link remains independently accessible.**
```
