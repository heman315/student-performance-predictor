import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample training data
data = {
    'Hours':[2,3,4,5,1,6,2.5,3.5,4.5,5.5,1.5,6.5,2,3,4,5],
    'Attendance':[70,80,90,95,60,100,75,85,90,95,65,100,70,80,90,95],
    'PrevMarks':[55,60,70,75,50,85,58,65,72,78,52,88,55,62,70,80],
    'Assignments':[3,4,5,5,2,6,3,4,5,6,2,6,3,4,5,5],
    'Sleep':[6,7,8,7,5,8,6,7,7,8,6,8,5,6,7,8],
    'Performance':['Low','Medium','Medium','High','Low','High','Medium','Medium','High','High','Low','High','Low','Medium','Medium','High']
}

df = pd.DataFrame(data)

# Features & target
X = df[['Hours','Attendance','PrevMarks','Assignments','Sleep']]
y = df['Performance']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")