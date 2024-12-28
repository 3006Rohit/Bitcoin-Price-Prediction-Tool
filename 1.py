import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Load Dataset
def load_data(ticker, start_date, end_date):
    """
    Load Bitcoin price data from Yahoo Finance API.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    # Flatten the MultiIndex columns
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

    # Print the structure of the DataFrame to ensure 'Close' exists
    print("Columns in the data:", data.columns)
    print("First few rows of the data:")
    print(data.head())

    return data





# Step 2: Preprocess Data
def preprocess_data(data):
    """
    Preprocess the data: handle missing values and add new features.
    """
    # Ensure the Close_BTC-USD column exists
    if 'Close_BTC-USD' not in data.columns:
        raise KeyError("'Close_BTC-USD' column not found in the dataset.")

    # Check if 'Close_BTC-USD' column contains numeric values
    data['Close_BTC-USD'] = pd.to_numeric(data['Close_BTC-USD'], errors='coerce')

    # Forward-fill missing values
    data.ffill(inplace=True)

    # Feature Engineering
    try:
        data['7_day_MA'] = data['Close_BTC-USD'].rolling(window=7).mean()
        data['14_day_MA'] = data['Close_BTC-USD'].rolling(window=14).mean()
        data['Volatility'] = data['High_BTC-USD'] - data['Low_BTC-USD']
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        print(data.head())
        raise

    # Drop rows with NaN values after feature engineering
    data.dropna(inplace=True)

    # Reset index after cleaning
    data.reset_index(drop=True, inplace=True)

    return data



# Step 3: Exploratory Data Analysis (EDA)
def visualize_data(data):
    """
    Visualize trends, distributions, and correlations in the dataset.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date_'], data['Close_BTC-USD'], label='Closing Price', color='blue')
    plt.title('Bitcoin Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    sns.heatmap(data[['Close_BTC-USD', '7_day_MA', '14_day_MA', 'Volatility']].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# Step 4: Train-Test Split
def split_data(data):
    """
    Split the data into training and test sets.
    """
    X = data[['7_day_MA', '14_day_MA', 'Volatility']]
    y = data['Close_BTC-USD']  # Update this line to use 'Close_BTC-USD'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Step 5: Train Model
def train_model(X_train, y_train):
    """
    Train an XGBoost regressor on the training data.
    """
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()

# Main Function
def main():
    ticker = 'BTC-USD'  # Bitcoin ticker symbol
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    
    # Load and preprocess the data
    data = load_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    # Visualize the data
    visualize_data(data)

    # Split data and train the model
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
