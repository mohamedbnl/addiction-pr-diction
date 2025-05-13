# Social Media Addiction Prediction

This project uses Machine Learning to predict potential addiction to social media platforms based on user behavior data. The goal is to identify patterns that correlate with time-wasting habits on social networks.

## 📁 Project Structure

- `ml_projet.ipynb`: Jupyter notebook containing the exploratory data analysis (EDA), preprocessing, model training and evaluation.
- `rf_addiction.pkl`: Trained Random Forest model saved using pickle.
- `adiction_app.py`: A simple web application (likely using Flask) to serve the model and allow predictions via a user interface.
- `Time-Wasters - Social Media.csv`: The dataset used for training and testing the model.

## 🧠 Machine Learning Model

- Algorithm: Random Forest Classifier
- Goal: Binary classification (Addicted vs Not Addicted)
- Features: Based on behavioral metrics like time spent, frequency of visits, etc.
