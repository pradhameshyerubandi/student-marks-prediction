import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Score': [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

model = LinearRegression()
model.fit(X, y)

predicted_score = model.predict(([[6.5]]))
print("Predicted score:", predicted_score[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Marks Prediction")
plt.savefig("student_marks_graph.png",dpi=300)
plt.show()