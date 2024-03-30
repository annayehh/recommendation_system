## Movie/Show Recommendation System

The objective of this project was to develop a sophisticated recommendation system for Netflix users, enhancing their content discovery experience. By employing advanced machine learning techniques, our project addresses the challenge of efficiently and effectively suggesting TV shows and movies based on users' preferences. We explored various algorithms, including Convolutional Neural Networks (CNN), K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). After careful evaluation, KNN emerged as the most suitable method for textual information-based recommendations. Our experimental results demonstrate a remarkable 41% accuracy in predicting IMDB scores. Furthermore, we offer insightful recommendations for enhancing accuracy, such as data augmentation and handling class imbalance. To provide users with a personalized recommendation system, we have implemented a Flask web application.

### Setup

To run the Flask app, please ensure the following libraries are installed:

```txt
Flask==2.3.2
Flask_Bootstrap==3.3.7.1
matplotlib==3.7.1
numpy==1.23.5
pandas==2.0.1
scikit_learn==1.2.2
seaborn==0.12.2
```

These dependencies are also listed in the `requirements.txt` file. To install all the dependencies, execute the command `pip install -r requirements.txt`.

Once all the necessary libraries are imported, launch the app by executing the following command: `python3 recommendation.py`.

This command will start the server for our web app, accessible at `http://localhost:5000/`.

### Demo

https://github.com/anish-kondepudi/ECS171_Group24_Project/assets/72046642/64381a9d-6b19-462d-bfc0-fbf80effe617
