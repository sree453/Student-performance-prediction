# 🎓 Student Performance Prediction using Decision Tree Classifier

## 📌 Project Overview

This project predicts whether a student will **Pass** or **Fail** based on academic, personal, family, and lifestyle-related factors using the **Decision Tree Classification** algorithm.

The model was built using Python and Scikit-learn and deployed using **Streamlit** to provide an interactive web application where users can enter student information and receive instant predictions.

---

# 📊 Problem Statement

Educational institutions often want to identify students who are at risk of failing so that appropriate guidance and support can be provided.

This project builds a machine learning model capable of predicting student performance based on various attributes such as:

* Academic performance
* Family background
* Study habits
* Lifestyle
* School support

---

# 🎯 Objective

The objective of this project is to:

* Predict whether a student will **Pass** or **Fail**
* Understand the important factors affecting student performance
* Learn the complete Decision Tree Classification workflow
* Deploy the trained model using Streamlit

---

# 📂 Dataset

**Dataset:** Student Performance Dataset

The dataset contains information about students including:

* School
* Gender
* Age
* Family Size
* Parent Education
* Parent Occupation
* Study Time
* Failures
* School Support
* Internet Access
* Health
* Absences
* First Period Grade (G1)
* Second Period Grade (G2)
* Final Grade (G3)

---

# 🎯 Target Variable

A new target column named **Result** was created using the Final Grade (G3).

**Logic:**

* G3 ≥ 10 → Pass
* G3 < 10 → Fail

After creating the target column, **G3 was removed** from the feature set to avoid **data leakage**.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle

---

# 📋 Project Workflow

## Step 1: Data Loading

* Loaded the Student Performance dataset using Pandas.

---

## Step 2: Exploratory Data Analysis (EDA)

Performed:

* Dataset inspection
* Shape of dataset
* Data types
* Statistical summary
* Missing value analysis
* Duplicate value analysis

---

## Step 3: Data Preprocessing

### Missing Values

Checked for missing values.

No missing values were found.

### Duplicate Records

Checked for duplicate rows.

Duplicate records were removed if present.

---

## Step 4: Target Creation

Created a new column:

**Result**

using the Final Grade (G3).

Pass → G3 ≥ 10

Fail → G3 < 10

---

## Step 5: Feature Selection

Dropped:

* G3

Selected remaining columns as input features.

---

## Step 6: Encoding

Categorical features were converted into numerical values using **Label Encoding**.

Separate LabelEncoders were created for each categorical feature.

The encoders were saved using Pickle for deployment.

---

## Step 7: Splitting Dataset

The dataset was divided into:

* Training Data (80%)
* Testing Data (20%)

using Train-Test Split.

---

## Step 8: Model Building

Implemented:

**Decision Tree Classifier**

Model Parameters:

* Criterion = Gini (and Entropy for comparison)
* Random State = 42

---

## Step 9: Model Training

The Decision Tree model was trained using the training dataset.

---

## Step 10: Model Prediction

Predictions were made on the testing dataset.

---

## Step 11: Model Evaluation

The following evaluation metrics were calculated:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## Step 12: Data Visualization

Visualizations created:

* Confusion Matrix Heatmap
* Decision Tree Visualization
* Feature Importance Analysis

---

## Step 13: Model Saving

The trained model was saved using Pickle.

Files generated:

* decision_tree_model.pkl
* encoders.pkl

---

## Step 14: Streamlit Deployment

Developed a Streamlit application that allows users to:

* Enter student information
* Predict student performance
* Display Pass/Fail prediction
* Show prediction confidence

---

# 📈 Features Used

* School
* Sex
* Age
* Address
* Family Size
* Parent Status
* Mother's Education
* Father's Education
* Mother's Job
* Father's Job
* Reason for Choosing School
* Guardian
* Travel Time
* Study Time
* Past Failures
* School Support
* Family Support
* Paid Classes
* Extracurricular Activities
* Nursery
* Higher Education
* Internet Access
* Romantic Relationship
* Family Relationship
* Free Time
* Going Out
* Weekday Alcohol Consumption
* Weekend Alcohol Consumption
* Health Status
* Absences
* G1
* G2

---

# 📊 Machine Learning Algorithm

Decision Tree Classifier

Decision Tree builds a tree-like structure by selecting the best feature at each node using impurity measures such as:

* Gini Index
* Entropy
* Information Gain

The model recursively splits the dataset until the leaf nodes become sufficiently pure.

---

# 📁 Project Structure

```
Student_Performance_Prediction/

│
├── app.py
├── decision_tree_model.pkl
├── encoders.pkl
├── student-mat.csv
├── requirements.txt
├── README.md
└── DecisionTree.ipynb
```

---

# ▶️ How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Random Forest Comparison
* XGBoost Comparison
* Feature Engineering
* Model Performance Optimization
* Cloud Deployment

---

# 📚 Learning Outcomes

Through this project, I learned:

* Decision Tree Classification
* Gini Index
* Entropy
* Information Gain
* Data Preprocessing
* Label Encoding
* Train-Test Split
* Model Evaluation
* Confusion Matrix
* Classification Report
* Feature Importance
* Model Serialization using Pickle
* Streamlit Deployment

---

# 👩‍💻 Author

**Sreeja Theegala**

This project was developed as part of my Machine Learning learning journey to strengthen my understanding of Decision Tree Classification and model deployment using Streamlit.
