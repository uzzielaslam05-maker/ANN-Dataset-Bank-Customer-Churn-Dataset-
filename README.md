# Bank Customer Churn Prediction Using Artificial Neural Network (ANN)

## Project Overview

This project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a bank customer is likely to leave the bank (churn) or remain a customer.

The project covers the complete Machine Learning pipeline including:

* Data Loading
* Data Exploration
* Data Preprocessing
* Feature Engineering
* Data Scaling
* ANN Model Building
* Model Training
* Model Evaluation
* Model Saving and Loading
* Performance Visualization

---

## Dataset Information

**Dataset:** Bank Customer Churn Dataset

### Features

| Feature          | Description                               |
| ---------------- | ----------------------------------------- |
| credit_score     | Customer credit score                     |
| country          | Customer country                          |
| gender           | Customer gender                           |
| age              | Customer age                              |
| tenure           | Number of years with bank                 |
| balance          | Account balance                           |
| products_number  | Number of products owned                  |
| credit_card      | Has credit card (0/1)                     |
| active_member    | Active member status (0/1)                |
| estimated_salary | Estimated salary                          |
| churn            | Target variable (0 = No Churn, 1 = Churn) |

### Target Variable

**churn**

* 0 → Customer stays
* 1 → Customer leaves

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Removed Unnecessary Columns

The `customer_id` column was removed because it acts only as an identifier and provides no predictive value.

### 2. Missing Value Handling

The dataset contained no missing values.

### 3. Categorical Encoding

One-Hot Encoding was applied to:

* country
* gender

Example:

Gender:

* Male
* Female

becomes:

* gender_Male
* gender_Female

### 4. Feature Scaling

StandardScaler was applied to normalize numerical features.

This improves ANN training performance and convergence speed.

---

## Model Architecture

Artificial Neural Network (ANN)

Input Layer

↓

Dense Layer (64 neurons, ReLU)

↓

Dropout (0.3)

↓

Dense Layer (32 neurons, ReLU)

↓

Dropout (0.3)

↓

Dense Layer (16 neurons, ReLU)

↓

Output Layer (1 neuron, Sigmoid)

### Activation Functions

**ReLU**

Used in hidden layers because it helps the network learn complex patterns efficiently.

**Sigmoid**

Used in the output layer because the problem is binary classification.

---

## Training Configuration

| Parameter        | Value               |
| ---------------- | ------------------- |
| Optimizer        | Adam                |
| Loss Function    | Binary Crossentropy |
| Batch Size       | 32                  |
| Epochs           | 100                 |
| Early Stopping   | Enabled             |
| Validation Split | 20%                 |

### Early Stopping

Training automatically stops when validation loss stops improving for 10 consecutive epochs.

This helps prevent overfitting.

---

## Evaluation Metrics

The following metrics were used:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures how many predicted churn customers actually churned.

### Recall

Measures how many actual churn customers were correctly identified.

### F1-Score

Balances Precision and Recall.

### ROC-AUC Score

Measures the model's ability to distinguish between churn and non-churn customers.

---

## Visualizations

The project includes:

* Churn Distribution Plot
* Training Loss Curve
* Validation Loss Curve
* Training Accuracy Curve
* Validation Accuracy Curve
* Confusion Matrix
* ROC Curve

---

## Model Saving

The trained model is saved as:

```python
model.save("bank_churn_ann.keras")
```

Model can be loaded later using:

```python
from tensorflow.keras.models import load_model

model = load_model("bank_churn_ann.keras")
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow / Keras

## Installation

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Structure

```text
Bank-Churn-ANN/
│
├── bank_churn.csv
├── bank_churn_notebook.ipynb
├── bank_churn_ann.keras
├── README.md
```

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Class imbalance handling
* Feature engineering
* XGBoost comparison
* Deployment using Streamlit or FastAPI

## Troubleshooting

### Dataset Not Found

Ensure:

```python
bank_churn.csv
```

is located in the same folder as the notebook.

### TensorFlow Not Installed

Run:

```bash
pip install tensorflow
```

### Shape Mismatch Error

Ensure prediction data contains the exact same columns used during training.

### GPU Issues

If TensorFlow cannot detect GPU, install the CPU version:

```bash
pip install tensorflow
```

## Author

Machine Learning Classification Project

Artificial Neural Network for Customer Churn Prediction

