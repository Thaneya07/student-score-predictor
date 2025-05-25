# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Create sample dataset
data = {'Hours': [1, 2, 3, 4.5, 5, 6.1, 7.2, 8.5, 9.6, 10],
        'Scores': [10, 20, 30, 41, 49, 60, 69, 81, 90, 95]}
df = pd.DataFrame(data)

# Step 3: Plot the data
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Study Hours vs Student Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid()
plt.show()

# Step 4: Split data
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Visualize regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')  # Regression line
plt.title('Regression Line on Study Data')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.grid()
plt.show()

# Step 8: Predict for custom input
hours = float(input("Enter study hours: "))
predicted_score = model.predict([[hours]])
print(f"Predicted Score: {predicted_score[0]:.2f}")
