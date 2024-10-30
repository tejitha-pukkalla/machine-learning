import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

data= pd.read_csv('profit_loss_data.csv', parse_dates=['Date'])

data['DayOfYear']= data['Date'].dt.day_of_year
x = data['DayOfYear'].values.reshape(-1,1)
y = data['Profit_Loss'].values

model = LinearRegression()
model.fit(x,y)

def predict_profit_loss(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date.timetuple().tm_yday
    predicted_value = model.predict(np.array([[day_of_year]]))
    return predicted_value[0]

user_date = '2024-11-28'
predicted_value = predict_profit_loss(user_date)

print(f"predicted Profit/Loss for {user_date}: {predicted_value}")