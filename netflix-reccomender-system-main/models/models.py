import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file
df = pd.read_csv('CleanTitles.csv')

# Select the desired attributes for prediction
attributes = ['tmdb_score', 'imdb_votes', 'runtime', 'release_year', 'seasons']
target = 'imdb_score'

# Filter the DataFrame to keep only the selected attributes and the target variable
df_filtered = df[attributes + [target]]

# Remove rows with missing values
df_filtered = df_filtered.dropna()

# Split the data into features (X) and target variable (y)
X = df_filtered[attributes]
y = df_filtered[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
knn = KNeighborsRegressor()
random_forest = RandomForestRegressor()
gradient_boosting = GradientBoostingRegressor()
support_vector = SVR()

# Fit the models
knn.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)
support_vector.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)
y_pred_gradient_boosting = gradient_boosting.predict(X_test)
y_pred_support_vector = support_vector.predict(X_test)

# Calculate MSE
mse_knn = mean_squared_error(y_test, y_pred_knn)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)
mse_support_vector = mean_squared_error(y_test, y_pred_support_vector)

# Calculate R-squared
r2_knn = r2_score(y_test, y_pred_knn)
r2_random_forest = r2_score(y_test, y_pred_random_forest)
r2_gradient_boosting = r2_score(y_test, y_pred_gradient_boosting)
r2_support_vector = r2_score(y_test, y_pred_support_vector)

# Print the results
print("K-Nearest Neighbors Regression:")
print("Mean Squared Error:", mse_knn)
print("R-squared:", r2_knn)
print()
print("Random Forest Regression:")
print("Mean Squared Error:", mse_random_forest)
print("R-squared:", r2_random_forest)
print()
print("Gradient Boosting Regression:")
print("Mean Squared Error:", mse_gradient_boosting)
print("R-squared:", r2_gradient_boosting)
print()
print("Support Vector Regression:")
print("Mean Squared Error:", mse_support_vector)
print("R-squared:", r2_support_vector)
