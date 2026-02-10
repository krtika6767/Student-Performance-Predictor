import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    data = pd.read_csv("student_data.csv")

    X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
    y = data['Final_Score']

    model = LinearRegression()
    model.fit(X, y)

    return model
