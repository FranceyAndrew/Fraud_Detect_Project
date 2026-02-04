# Credit Card Fraud Detection - Machine Learning Project

![Python 3.12.3 Badge](https://img.shields.io/badge/Python-3.12.3-blue.svg)
![Jupyter Notebook Badge](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Panda Data Badge](https://img.shields.io/badge/Panda-Data%20Analysis-yellow.svg)
![Scikit Badge](https://img.shields.io/badge/Scikit--learn-ML-green.svg)
![Status Active success Badge](https://img.shields.io/badge/Status-Active-success.svg)
![License MIT Badge](https://img.shields.io/badge/License-MIT-lightgrey.svg)

This project builds an end-to-end fraud detection pipeline using the Kartik2112 Fraud detection dataset, a rich synthetic dataset containing demographic, geographic, temporal, and transactional inforamation. The goal is to explore the data, engineer meaningful features, and train machine mearning models capable of identifying fraudulent transactions in a highly imbalanced environment.

## 📂 Project Structure

* project/
    * data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Raw and processed datasets (not tracked in Git)
    * notebooks/
        * 01_eda.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Exploratory Data Analysis
        * 02_preprocessing.ipynb&nbsp;&nbsp;&nbsp;&nbsp;# Feature engineering + cleaning
        * 03_modeling.ipynb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Model training + evaluation
    * src/
        * data_loader.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Data loading utility
        * preprocessing.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Prepocessing + feature engineering pipeline
        * model_utils.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Training, evaluation, and helper functions
    * README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Project documentation
    * LICENSE
    * requirements&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dependences needed to run project 


## 📊 Dataset Overview

The project uses the Kartik2112 Credit Card Transactions Fraud Detection Dataset can be found at https://www.kaggle.com/datasets/kartik2112/fraud-detection.

The dataset includes the following features:
* Transaction details (amount, merchant, category, timestamp, credit card number)
* Customer demographics (date of birth, job, first name, last name, gender, location)
* Merchant information (merchant name, category, location)
* Temporal features (Unix_time)
* Fraud label (is_fraud)

## 🔍 Exploratory Data Analysis (EDA)

Key insights from the EDA:
* Fraud accounts for a very small percentage of all transactions.
* No single feature strongly correlates with fraud - typical for fraud datasets.
* Geographic features show strong internal correlations, suggesting value in distance-based features.
* Time-derived features (hour, weekday, month) show possible behavioral patterns.
* Transaction amount distributions differ between fraud and non-fraud cases. 
* Some states have extremely low sample counts. Some transactions were removed to avoid leakage.

The EDA notebook includes:
* Fraud distribution analysis 
* Amount distributions
* Temporal patterns
* Geographic and demographic exploration
* Correlation heatmap
* Fraud-rate visualizations