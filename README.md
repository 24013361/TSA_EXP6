# Ex.No: 6               HOLT WINTERS METHOD
### Date: 19-05-2026
### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv('chennai_temperature_10years.csv',
    parse_dates=['Date'],
    index_col='Date'
)

# Preview dataset
print("\nFirst 5 Rows:")
print(data.reset_index().head())

# Sort by date
data = data.sort_index()

# Resample monthly average temperature
data_monthly = data['Temperature'].resample('ME').mean()

# Plot monthly data
plt.figure(figsize=(10,5))
data_monthly.plot()
plt.title("Monthly Average Chennai Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.grid(True)
plt.show()

# Scaling the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly.values.reshape(-1, 1)
    ).flatten(),
    index=data_monthly.index
)

# Plot scaled data
plt.figure(figsize=(10,5))
scaled_data.plot()
plt.title("Scaled Temperature Data")
plt.grid(True)
plt.show()

# Add 1 for multiplicative seasonality
scaled_data = scaled_data + 1

# Seasonal decomposition
decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

# Plot decomposition
decomposition.plot()
plt.show()

# Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Holt Winters Model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast on test data
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# Visual evaluation
plt.figure(figsize=(12,6))

ax = train_data.plot(label='train_data')
test_predictions_add.plot(ax=ax, label='test_predictions_add')
test_data.plot(ax=ax, label='test_data')

ax.legend()
ax.set_title('Visual Evaluation')
plt.grid(True)
plt.show()

# RMSE
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

print("\nRMSE:")
print(rmse)

# Standard deviation and mean
print("\nStandard Deviation and Mean:")
print(
    np.sqrt(scaled_data.var()),
    scaled_data.mean()
)

# Final model
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Future prediction
final_predictions = final_model.forecast(
    steps=int(len(data_monthly) / 4)
)

# Final prediction plot
plt.figure(figsize=(12,6))

ax = scaled_data.plot(label='data_monthly')
final_predictions.plot(ax=ax, label='final_predictions')

ax.legend()
ax.set_title('Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Temperature')

plt.grid(True)
plt.show()
```
### OUTPUT:
<img width="674" height="376" alt="image" src="https://github.com/user-attachments/assets/60246571-9cfc-450d-8d21-eb04e1f8114b" />

<img width="663" height="390" alt="image" src="https://github.com/user-attachments/assets/27011dda-6559-4c34-8247-ff94aaada9dc" />


<img width="948" height="472" alt="image" src="https://github.com/user-attachments/assets/1d266a41-ce2b-4405-ad5e-5f0992382c60" />

TEST_PREDICTION

<img width="793" height="524" alt="image" src="https://github.com/user-attachments/assets/8dd1628a-053c-4c6f-be3f-6ff86b207196" />


FINAL_PREDICTION
<img width="810" height="508" alt="image" src="https://github.com/user-attachments/assets/39450c12-40a4-4c60-8107-180e87519ef1" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
