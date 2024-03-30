import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('titles.csv')

# Exclude non-numeric columns
numeric_columns = data.select_dtypes(include=np.number).columns
numeric_data = data[numeric_columns]

# Extract unique genres
unique_genres = set()
for genres_list in data['genres']:
    genres = ast.literal_eval(genres_list)
    unique_genres.update(genres)

# One-hot encode genres (we ignore countries since there are too many)
genre_columns = []
for genre in unique_genres:
    genre_columns.append('genre_' + genre)
    data['genre_' + genre] = data['genres'].apply(lambda x: int(genre in ast.literal_eval(x)))

# Exclude original list column
data = data.drop(['genres', 'production_countries'], axis=1)

# Concatenate numeric and encoded non-numeric data
processed_data = pd.concat([numeric_data] + [data[genre_columns]], axis=1)

# Calculate the correlation matrix
correlation_matrix = processed_data.corr().round(3)  # Round to three decimal points

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f")  # Display three decimal points
plt.title('Correlation Matrix')
plt.show()

# Find the attributes with highest correlation to imdb_score
correlation_with_imdb_score = correlation_matrix['imdb_score'].abs().sort_values(ascending=False)
print(correlation_with_imdb_score)

# OUTPUT - SORTED BY MOST TO LEAST CORRELATED
# imdb_score             1.000
# tmdb_score             0.533
# genre_documentation    0.185
# imdb_votes             0.175
# runtime                0.168
# release_year           0.128
# genre_horror           0.118
# genre_history          0.117
# seasons                0.102
# genre_war              0.087
# genre_drama            0.086
# genre_thriller         0.076
# genre_comedy           0.075
# tmdb_popularity        0.073
# genre_animation        0.069
# genre_crime            0.046
# genre_family           0.044
# genre_action           0.042
# genre_romance          0.041
# genre_sport            0.037
# genre_reality          0.025
# genre_fantasy          0.022
# genre_western          0.017
# genre_music            0.012
# genre_scifi            0.005
# genre_european         0.001