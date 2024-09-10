# Bike Rental Demand Prediction

This project focuses on predicting hourly demand for bike rentals using data analytical and machine learning tools. The goal is to create a model that can forecast future demand based on various input features such as time, weather, and temperature. The repository contains the entire pipeline from data preprocessing, exploratory data analysis (EDA), and feature engineering to model development and evaluation.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset Information](#dataset-information)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Modeling Approaches](#modeling-approaches)
6. [Model Evaluation](#model-evaluation)
7. [Usage](#usage)
8. [Repository Structure](#repository-structure)
---

## Project Objective

The primary objective is to model the hourly demand for bike rentals, which can be used to predict bike demand for a future date. The target variables include:
- Casual Rides (`casual`)
- Registered Rides (`registered`)
- Total Rides (`count`)

---

## Dataset Information

The dataset is split into two categories:
- **Training data**: 10,886 examples.
- **Test data**: 6,494 examples.

**Input Features**:
- `datetime`: Timestamp up to the hour.
- `season`, `weather`: Categorical data about season and weather.
- `holiday`, `workingday`: Binary flags.
- `temp`, `atemp`, `humidity`, `windspeed`: Continuous variables representing weather conditions.

**Target Variables**:
- `casual`: Number of casual riders.
- `registered`: Number of registered riders.
- `count`: Total rides (sum of `casual` and `registered`).

---

## Exploratory Data Analysis

### 1. **Data Imbalance**
- **Seasons**: Balanced distribution across different seasons.
- **Holidays**: Dominated by observations on holidays.
- **Working Days**: Majority of observations are on working days.
- **Seasonal Variation**: Low observations for season 4 compared to others.

### 2. **Variation Based on Categorical Variables**
- Significant variation in total rides based on **season** and **weather**.
- Negligible variation based on **holiday** and **workingday**.
  
### 3. **Correlation Analysis**
- **temp** and **atemp** show a strong positive correlation with total rides, particularly for casual rides.
- **Humidity**: Moderately negative correlation with total rides.
- **Windspeed**: Low correlation with target variables.

## Observations
- **Key Features**: `temp`, `humidity`, `season`, and `weather` are strong predictors of demand.
- **Redundancy**: High correlation between `temp` and `atemp` (0.98), so only one should be used.

---

## Feature Engineering

We created additional features based on the `datetime` variable:
- **Hour of the Day**: Captures peak hours.
- **Day of the Week**: Captures differences in demand based on the day.
- **Week of the Month**: Captures demand variation within the month.
- **Month of the Year**: Captures seasonal demand.
- **Year**: Captures changes in demand trends over the years.

### Observations:
- **Hour of the Day**: High impact on total rides.
- **Month of the Year** and **Year**: Strong impact on total rides.
- **Week of the Month** and **Day of the Week**: Lower direct impact.

---

## Modeling Approaches

We used three different approaches to model the target variables:
1. **Separate Single-Output Models**: Independent models trained for `casual`, `registered`, and `count`.
2. **Multi-Output Model**: A single model trained to predict all three target variables simultaneously.
3. **Derived Model**: Separate models for `casual` and `registered`, and the `count` is derived by summing both predictions.

**Models Used**:
- **Linear Regression**: Used as a baseline.
- **Random Forest Regressor (RF)**: Fitted with all approaches.
- **XGBoost Regressor (XGB)**: Also fitted with all approaches, and hyperparameters were fine-tuned using `GridSearchCV`.

---

## Model Evaluation

After fitting all models, XGBoost performed the best, with the following observations:

- **XGBoost** outperforms Linear Regression and Random Forest, with the highest R2 score of **0.96**.
- **Best Approach**: Derived model (separate models for `casual` and `registered` rides, summing their predictions to get `count`) showed the lowest MAE of **23.17** for `count`.
- **Feature Importance**: Time-based features like `hour`, `year`, and `workingday` were found to be the most significant.

### Benchmark for Model Evaluation
The test data includes three target variables: count, registered, and casual. The model's performance is evaluated in the following order of significance:

1. **Primary Objective:** Maximize the R² score and minimize MSE for the count variable.
2. **Secondary Objective:** Maximize the R² score and minimize MSE for the registered and casual variables.

The **primary benchmark** for model performance is the R² score on the count variable. Secondary benchmarks are the R² scores on the registered and casual variables. The model with the best R² score for count and competitive R² scores for the other two variables will be considered the best-performing model

| Model      | Approach     | R2 (Count) | R2 (Casual) | R2 (Registered) |
|------------|--------------|-------------|-------------|-----------------|
| Linear Regression | Single-Output | 0.393 | 0.481     | 0.335           |
| XGBoost    | Single-Output | **0.958**  | **0.928**   | **0.957**       |
| XGBoost    | Multi-Output | **0.958**   | **0.928**   | **0.957**       |
| XGBoost    | Derived      | **0.958**   | **0.928**   | **0.957**       |
| RandomForest | Single-Output | 0.944    | 0.911       | 0.947           |
| RandomForest | Multi-Output | 0.944     | 0.911       | 0.947           |
| RandomForest | Derived    | 0.944       | 0.94        | 0.93            |

---

## Usage

To clone and run this repository locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/bike-rental-demand-prediction.git
    cd bike-rental-demand-prediction
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows use: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the main.py file to predict bike demand**:
    ```bash
    python main.py
    ```

5. **Test Predictions**: The `test_predictions.csv` file contains the model predictions on the test data.

---

### Repository Structure

```
├── data/
│   ├── train.csv
│   └── test.csv
├── Analysis/
│   ├── Bike Demand Prediction.ipynb
│   ├── model_comparison_metrics.xlsx
│   ├── xgb_casual_model.pkl
│   └── xgb_registered_model.pkl
├── requirements.txt
├── test_predictions.csv
├── main.py
└── README.md
```

- `data/`: Contains the training and test datasets.
- `Analysis/`: Contains the Jupyter notebook for EDA and model comparison results, as well as the saved models.
- `requirements.txt`: List of dependencies to install.
- `main.py`: The main script to generate predictions from the models.
- `test_predictions.csv`: Contains the predictions on the test dataset.

---

Feel free to clone the repository and explore the project!