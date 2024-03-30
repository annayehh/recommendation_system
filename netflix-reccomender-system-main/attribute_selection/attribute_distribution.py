import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv('titles.csv')

# Select the attributes of interest
selected_attributes = ['tmdb_score', 'imdb_votes', 'runtime', 'release_year']

# Create a single figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Plot scatter plots and regression lines for each attribute
for i, attribute in enumerate(selected_attributes):
    row = i // 2
    col = i % 2
    ax = axes[row][col]
    sns.regplot(x=attribute, y='imdb_score', data=data, scatter_kws={'alpha': 0.5}, ax=ax)
    ax.set_xlabel(attribute)
    ax.set_ylabel('imdb_score')
    ax.set_title(f'{attribute} vs. imdb_score')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
