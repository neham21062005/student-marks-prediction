import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [1,2,3,4,5,6,7,8],
    'Marks': [10,20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[5]])

print("Predicted marks:", prediction[0])
