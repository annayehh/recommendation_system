from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)
bootstrap = Bootstrap(app)

def get_model():
    static_folder = os.path.join(app.root_path, 'static')
    csv_file_path = os.path.join(static_folder, 'CleanTitles.csv')
    df = pd.read_csv(csv_file_path)

    attributes = ['tmdb_score', 'imdb_votes', 'runtime', 'release_year', 'seasons']
    target = 'imdb_score'

    df_filtered = df[attributes + [target]]
    df_filtered = df_filtered.dropna()

    X = df_filtered[attributes]
    y = df_filtered[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gradient_boosting = GradientBoostingRegressor()
    gradient_boosting.fit(X_train, y_train)

    return gradient_boosting

model = get_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        tmdb_score = float(request.form.get('tmdb_score'))
        imdb_votes = int(request.form.get('imdb_votes'))
        runtime = int(request.form.get('runtime'))
        release_year = int(request.form.get('release_year'))
        seasons = int(request.form.get('seasons'))

        data = {
            'tmdb_score': [tmdb_score],
            'imdb_votes': [imdb_votes],
            'runtime': [runtime],
            'release_year': [release_year],
            'seasons': [seasons]
        }
        df = pd.DataFrame(data)
        imdb_score_pred = model.predict(df)[0]

        result = f"Predicted IMDB score: {str(round(imdb_score_pred, 2))}"

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
