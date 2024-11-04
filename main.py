import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('profit_loss_data.csv', parse_dates=['Date'])
    data['DayOfYear'] = data['Date'].dt.day_of_year
    x = data['DayOfYear'].values.reshape(-1, 1)
    y = data['Profit_Loss'].values
except (FileNotFoundError, pd.errors.ParserError, KeyError) as e:
    print(f"Error loading or processing data: {e}")
    exit()

try:
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(x)
except ValueError as e:
    print(f"Error in clustering: {e}")
    exit()


try:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(x, y)
    linear_model = LinearRegression().fit(x, y)
except ValueError as e:
    print(f"Error training model: {e}")
    exit()


def predict_profit_loss(model, date_str):
    try:
        day_of_year = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
        return model.predict([[day_of_year]])[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


user_date = '2024-11-28'
print(f"Random Forest Prediction for {user_date}: {predict_profit_loss(rf_model, user_date)}")
print(f"Linear Regression Prediction for {user_date}: {predict_profit_loss(linear_model, user_date)}")


try:
    x_range = np.arange(1, 366).reshape(-1, 1)
    y_pred_linear = linear_model.predict(x_range)
    y_pred_rf = rf_model.predict(x_range)

    plt.figure(figsize=(12, 6))
    
    
    for cluster in range(3):
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data['DayOfYear'], cluster_data['Profit_Loss'], label=f'Cluster {cluster}')
    
    
    plt.plot(x_range, y_pred_linear, color='red', label='Linear Regression Line')
    plt.plot(x_range, y_pred_rf, color='green', label='Random Forest Regression Line')
    
    plt.xlabel('Day of Year')
    plt.ylabel('Profit/Loss')
    plt.title('Profit/Loss Prediction and Clustering')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")