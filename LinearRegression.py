from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
X = np.array([10, 15, 20, 30, 35, 20, 25, 40]).reshape(-1, 1)
Y = np.array([50, 70, 90, 150, 170, 110, 140, 220])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y)

# Print the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Calculate predictions
predictions = model.predict(X)

# Calculate the least squares
least_squares = np.sum((predictions - Y) ** 2)
print("Least squares:", least_squares)


# Plot the dataset
plt.scatter(X, Y, color='blue', label='Original data')

# Plot the linear regression line
plt.plot(X, predictions, color='red', label='Linear regression')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()

# Show plot
plt.grid(True)
plt.show()