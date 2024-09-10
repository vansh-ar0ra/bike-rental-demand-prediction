# Load necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler


# Load train & test data
train_data = pd.read_csv('../data/train.csv', parse_dates=['datetime'])
train_data.set_index('datetime', inplace=True)

test_data = pd.read_csv('../data/test.csv', parse_dates=['datetime'])
test_data.set_index('datetime', inplace=True)


# One-Hot Encoding
# train
train_data_encoded = pd.get_dummies(train_data, columns=[
    'season', 'holiday', 'workingday', 'weather'], drop_first=True)

# test
test_data_encoded = pd.get_dummies(test_data, columns=[
    'season', 'holiday', 'workingday', 'weather'], drop_first=True)

# Extract time features
# train
train_data_encoded['hour'] = train_data.index.hour
train_data_encoded['day_of_week'] = train_data.index.dayofweek
train_data_encoded['month'] = train_data.index.month
train_data_encoded['year'] = train_data.index.year
train_data_encoded['day_of_month'] = train_data.index.day
train_data_encoded['week_of_month'] = train_data_encoded.day_of_month.apply(
    lambda x: (x-1)//7 + 1
)
train_data_encoded.drop(['atemp', 'day_of_month'], axis=1, inplace=True)

# test
test_data_encoded['hour'] = test_data.index.hour
test_data_encoded['day_of_week'] = test_data.index.dayofweek
test_data_encoded['month'] = test_data.index.month
test_data_encoded['year'] = test_data.index.year
test_data_encoded['day_of_month'] = test_data.index.day
test_data_encoded['week_of_month'] = test_data_encoded.day_of_month.apply(
    lambda x: (x-1)//7 + 1
)
test_data_encoded.drop(['atemp', 'day_of_month'], axis=1, inplace=True)


# Initialize & fit the scaler on numerical features

# train
numerical_features = ['temp', 'humidity', 'windspeed', 'hour', 'day_of_week', 'month', 'year', 'week_of_month']
scaler = StandardScaler()
scaler.fit_transform(train_data_encoded[numerical_features])

# test
test_data_encoded[numerical_features] = scaler.transform(
    test_data_encoded[numerical_features])


# No need to process training data after this step

# Load or define the multi-output XGBoost model
xgb_causal_model = joblib.load('xgb_casual_model.pkl')
xgb_reg_model = joblib.load('xgb_registered_model.pkl')

# Predict using the trained multi-output model
predictions_causal = xgb_causal_model.predict(test_data_encoded)
predictions_registered = xgb_reg_model.predict(test_data_encoded)
predictions_total = predictions_causal + predictions_registered

# Combine predictions into a single DataFrame
predictions_multi_output = np.column_stack(
    (predictions_causal, predictions_registered, predictions_total))

# Converting negative preds to 0
predictions_multi_output[predictions_multi_output < 0] = 0

# Convert predictions to a DataFrame
predictions = pd.DataFrame(predictions_multi_output, columns=[
                           'casual', 'registered', 'count'], index=test_data_encoded.index)

# Wrtie predictions to csv
predictions.to_csv('../test_predictions.csv')