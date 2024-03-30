import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# Load the data from the CSV file
data = pd.read_csv("CleanTitles.csv")

# Select the features (attributes) and the target variable
features = ["tmdb_score", "imdb_votes", "runtime", "release_year", "seasons"]
target = "imdb_score"

# Drop rows with missing values in the selected features or target variable
data = data.dropna(subset=features + [target])

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7],
    'leaf_size': [10, 30, 50],
    'p': [1, 2]
}

# Create the KNN regressor
knn = KNeighborsRegressor()

# Perform grid search
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print best hyperparameters
print("Best Hyperparameters:")
print(best_params)

# Calculate and print the confusion matrix
y_pred_rounded = y_pred.round()
confusion = confusion_matrix(y_test.round(), y_pred_rounded)
print("Confusion Matrix:")
print(confusion)

# Print classification report
report = classification_report(y_test.round(), y_pred_rounded)
print("Classification Report:")
print(report)
