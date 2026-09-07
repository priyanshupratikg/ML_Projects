# 🩺 DiabDetect – Diabetes Risk Prediction System

### Machine Learning Based Diabetes Risk Prediction Web Application

DiabDetect is a machine-learning-based diabetes risk prediction system developed as an academic project. The application analyzes clinical measurements provided by a user and predicts the likelihood of diabetes using a trained and tuned Random Forest Classification model.

The project combines:

- Data preprocessing
- Missing-value treatment
- Feature scaling
- Machine learning model training
- Hyperparameter tuning
- Model evaluation
- Model serialization
- Streamlit web application
- Responsive desktop and mobile interface

---

## 🌐 Live Application

### 💻 PC / Laptop

👉 **[Open DiabDetect on PC](https://obscurity-hardly-wanted.ngrok-free.dev)**

### 📱 Mobile Phone

👉 **[Open DiabDetect on Mobile](https://obscurity-hardly-wanted.ngrok-free.dev)**

> **Note:** Both links open the same responsive web application. The interface automatically adapts to desktop, tablet and mobile screen sizes.

> ⚠️ **Important:** The live link is provided through an ngrok tunnel. It will only work while the Streamlit application and ngrok tunnel are running on the host computer. The URL may change when a new ngrok session is started.

---

# 📌 Project Overview

Diabetes is a chronic condition that requires early identification and proper medical attention. Machine learning can be used to analyze clinical measurements and identify patterns associated with diabetes.

The objective of this project is to develop an end-to-end machine learning system that:

1. Loads the diabetes dataset.
2. Performs data preprocessing.
3. Handles suspicious zero values.
4. Performs missing-value imputation.
5. Scales numerical features.
6. Trains multiple machine learning models.
7. Compares model performance.
8. Tunes the selected model.
9. Saves the trained model and preprocessing objects.
10. Integrates the final model into a Streamlit application.
11. Provides an interactive diabetes-risk prediction interface.

---

# 🎯 Objectives

The main objectives of the project are:

- To understand and preprocess a real-world medical dataset.
- To identify invalid or suspicious values.
- To handle missing values appropriately.
- To normalize/standardize numerical features.
- To train different machine learning algorithms.
- To compare their performance.
- To select and tune an appropriate final model.
- To save the trained model for deployment.
- To build an interactive prediction application.
- To provide a responsive interface usable on both PC and mobile devices.

---

# 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains clinical measurements that can be used to predict whether a patient belongs to the diabetic or non-diabetic class.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Patient age |
| Outcome | Diabetes class (0 or 1) |

---

# 🔬 Data Preprocessing

Data preprocessing was performed before training the machine learning models.

## 1. Suspicious Zero Values

Some features in the dataset cannot realistically have a value of zero.

For example:

- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI

Therefore, suspicious zero values were treated as missing values rather than valid measurements.

---

## 2. Missing Value Imputation

After identifying suspicious zero values, the missing values were handled using **median imputation**.

Median imputation was selected because medical datasets can contain extreme values, and the median is less affected by outliers than the mean.

The trained imputer was saved as:

```text
models/median_imputer.pkl
````

---

## 3. Feature Scaling

Numerical features were standardized using **StandardScaler**.

Standardization transforms the features so that they are on a comparable scale.

The fitted scaler was saved as:

```text
models/standard_scaler.pkl
```

---

# 🤖 Machine Learning

Multiple machine learning algorithms were considered and evaluated during the project.

The objective was to compare different approaches and select a suitable model for the final application.

The final deployed model is a:

## 🌲 Tuned Random Forest Classifier

Random Forest was selected as the final model after model comparison and tuning.

The trained model was saved as:

```text
models/final_random_forest.pkl
```

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed to improve the performance of the Random Forest classifier.

Important Random Forest parameters include:

* Number of estimators
* Maximum depth
* Minimum samples required for splitting
* Minimum samples required at a leaf
* Feature selection strategy

The final application uses the tuned Random Forest model rather than training a new model every time a prediction is requested.

---

# 🧠 Prediction Pipeline

The deployed application follows the following pipeline:

```text
User Input
     ↓
Input Validation
     ↓
Feature Preparation
     ↓
Missing Value Handling
     ↓
Standard Scaling
     ↓
Trained Random Forest Model
     ↓
Prediction Probability
     ↓
Risk Classification
     ↓
Result Display
```

---

# 🖥️ Web Application

The user interface was developed using **Streamlit**.

The application provides an interactive interface where users can enter patient information and clinical measurements.

### Patient Information

The application accepts:

* Patient Name
* Gender

### Clinical Measurements

The application accepts:

* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Age

### Pregnancy Handling

The number of pregnancies is displayed only when the selected gender is **Female**.

For male users, the pregnancy field is automatically excluded from the interface and handled appropriately by the application.

---

# 📏 Input Validation

The application provides the expected dataset ranges below the relevant input fields.

This helps users understand the acceptable range of values before submitting the prediction.

The application also performs validation to prevent clearly invalid inputs from being submitted.

Examples include:

* Negative clinical measurements
* Values outside the supported range
* Missing required patient information
* Invalid age
* Invalid pregnancy count

---

# 📈 Prediction Output

After the user submits the clinical measurements, the application displays:

* Patient name
* Predicted diabetes-risk class
* Estimated probability
* Prediction explanation

Example:

```text
Prediction Result

Patient: Example Patient

Higher Diabetes Risk

Estimated diabetes probability: 81.91%
```

The probability is generated by the trained Random Forest classifier.

---

# 🧮 Diabetes Pedigree Function

The original dataset contains the feature:

```text
DiabetesPedigreeFunction
```

However, this is not a practical value for a normal user to manually calculate.

Therefore, the application interface does **not require the user to enter the Diabetes Pedigree Function manually**.

The application handles the feature internally according to the project's prediction pipeline.

This keeps the interface simpler and more user-friendly.

---

# 📁 Project Structure

```text
Assignment1_Diabetes_prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── final_random_forest.pkl
│   ├── median_imputer.pkl
│   └── standard_scaler.pkl
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── results/
│
├── visualizations/
│
├── .gitignore
│
├── README.md
│
└── requirements.txt
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

## Development Tools

* Visual Studio Code
* Jupyter Notebook
* Git
* GitHub
* ngrok

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/priyanshupratikg/ML_Projects.git
```

Move into the project directory:

```bash
cd ML_Projects/Assignment1_Diabetes_prediction
```

---

# 🐍 Create Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

---

# 📚 Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

From the project directory, run:

```bash
streamlit run app/app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 🌍 Access From Another Device

To make the application temporarily accessible from another device, an ngrok tunnel can be used.

First start Streamlit:

```bash
streamlit run app/app.py
```

Then open another terminal and run:

```bash
ngrok http 8501
```

ngrok will generate a public forwarding URL similar to:

```text
https://example.ngrok-free.dev
```

Open that URL on:

* PC
* Laptop
* Android phone
* iPhone
* Tablet

provided the ngrok session and Streamlit application are still running.

---

# 📱 Responsive Design

The application interface has been designed to work across different screen sizes.

### Desktop

The desktop version provides:

* Sidebar navigation
* Two-column clinical measurement layout
* Detailed prediction result
* Full-width controls

### Mobile

The responsive layout adapts the interface for smaller screens.

Users can access the application through a mobile browser without installing a separate application.

Supported devices include:

* Android smartphones
* iPhones
* Tablets
* Laptops
* Desktop computers

---

# 🔐 Privacy & Security

This project is intended for academic and demonstration purposes.

Users should not enter sensitive personal medical information into a publicly accessible demo deployment.

The public ngrok link should be considered a temporary demonstration endpoint rather than a production medical service.

---

# ⚠️ Important Medical Disclaimer

**DiabDetect is an academic machine learning project and is NOT a medical diagnostic system.**

The prediction generated by this application should not be considered a medical diagnosis.

The application should not be used to:

* Diagnose diabetes
* Replace a doctor
* Replace laboratory testing
* Make medical treatment decisions
* Determine medication or dosage

For real medical concerns, users should consult a qualified healthcare professional.

---

# 📚 Academic Purpose

This project demonstrates an end-to-end machine learning workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Missing Value Treatment
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

# 👨‍💻 Author

### Priyanshu Pratik

Machine Learning Project
MLPWP – 4th Year

---

# 🔗 Project Links

### GitHub Repository

👉 [https://github.com/priyanshupratikg/ML_Projects](https://github.com/priyanshupratikg/ML_Projects)

### Diabetes Prediction Application – PC

👉 [https://obscurity-hardly-wanted.ngrok-free.dev](https://obscurity-hardly-wanted.ngrok-free.dev)

### Diabetes Prediction Application – Mobile

👉 [https://obscurity-hardly-wanted.ngrok-free.dev](https://obscurity-hardly-wanted.ngrok-free.dev)

---

# ⭐ Project Highlights

* End-to-end machine learning project
* Real-world diabetes dataset
* Data preprocessing pipeline
* Suspicious-zero handling
* Median imputation
* Standard scaling
* Multiple ML model comparison
* Tuned Random Forest classifier
* Serialized ML model
* Interactive Streamlit interface
* Input validation
* Gender-aware pregnancy field
* Responsive PC and mobile interface
* GitHub version control
* Temporary public deployment using ngrok

---

## 📌 Status

**Project Status: Completed ✅**

The machine learning pipeline and interactive Streamlit application have been implemented and integrated into a complete diabetes-risk prediction system.

```

### One correction I recommend before putting this README on GitHub

There is one technically important point in the README above: **don't claim that DPF is mathematically calculated from the patient's other inputs unless your `app.py` actually implements such a calculation.** The original Diabetes Pedigree Function is a dataset feature based on family-history information; it isn't normally something that can be reliably derived from glucose/BMI/age/etc.

Since your current app is working without asking the user for DPF, the README should describe exactly what your app does rather than imply a medical calculation that doesn't exist.

If your current `app.py` uses a **fixed/internal DPF value**, we should document that honestly (and ideally improve the model pipeline later).
```
