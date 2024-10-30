import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('profit_loss_data.csv', parse_dates=['Date'])

data['DayOfYear'] = data['Date'].dt.day_of_year
x = data['DayOfYear'].values.reshape(-1, 1)
y = data['Profit_Loss'].values

model = LinearRegression()
model.fit(x, y)

def predict_profit_loss(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date.timetuple().tm_yday
    predicted_value = model.predict(np.array([[day_of_year]]))
    return predicted_value[0]

x_range = np.arange(1, 366).reshape(-1, 1) 
y_pred = model.predict(x_range)

plt.figure(figsize=(10, 6))
plt.scatter(data['DayOfYear'], y, color='blue', label='Actual Profit/Loss')
plt.plot(x_range, y_pred, color='red', label='Regression Line')
plt.xlabel('Day of Year')
plt.ylabel('Profit/Loss')
plt.title('Profit/Loss Regression Over the Year')
plt.legend()
plt.show()
