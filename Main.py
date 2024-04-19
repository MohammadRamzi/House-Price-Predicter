import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Extract features (size) and target variable (price)
X = data['size'].values.reshape(-1, 1)
y = data['price'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict price based on size
def predict_price(size):
    price = model.predict(np.array(size).reshape(-1, 1))
    return price[0]

# Main function
def main():
    # User input for size
    size_input = float(input("Enter the size of the house (in square feet): "))
    
    # Predict price
    predicted_price = predict_price(size_input)
    print(f"Predicted price for a house with size {size_input} sqft: ${predicted_price:.2f}")

    # Visualize the dataset and regression line
    plt.scatter(X, y, color='blue', label='Original data')
    plt.plot(X, model.predict(X), color='red', label='Linear regression')
    plt.xlabel('Size (sqft)')
    plt.ylabel('Price ($)')
    plt.title('House Prices Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()