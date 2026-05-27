import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

st.title("🎓 Student Marks Prediction")

df = pd.read_csv("student_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

X = df[['study_hours',
        'attendance',
        'sleep_hours',
        'previous_marks']]

y = df['final_marks']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

prediction_test = model.predict(X_test)

mae = mean_absolute_error(y_test,prediction_test)
r2 = r2_score(y_test,prediction_test)

st.subheader("Model Performance")

st.write("MAE:",round(mae,2))
st.write("R² Score:",round(r2,2))

st.subheader("Enter Student Details")

hours = st.slider(
    "Study Hours",
    1,
    12,
    5
)

attendance = st.slider(
    "Attendance %",
    50,
    100,
    85
)

sleep = st.slider(
    "Sleep Hours",
    4,
    10,
    7
)

previous = st.slider(
    "Previous Marks",
    0,
    100,
    60
)

if st.button("Predict Marks"):

    result = model.predict(
        [[hours,
          attendance,
          sleep,
          previous]]
    )

    st.success(
        f"Predicted Marks: {result[0]:.2f}"
    )

st.subheader("Study Hours vs Final Marks")

fig,ax = plt.subplots()

ax.scatter(
    df['study_hours'],
    df['final_marks']
)

ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Marks")

st.pyplot(fig)
