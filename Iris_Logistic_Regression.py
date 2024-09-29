import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
Iris_data = df.to_numpy()

# Multiclass classification
# 1. Iris_data 를 input 과 target 으로 나누기
Iris_input = Iris_data[:,:2]
Iris_target = Iris_data[:,4]

# 2. input 과 target 을 train_set 과 test_set 으로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(Iris_input,Iris_target)
# train_set : test_set = 111:38

# 3. Data Preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 4. Model fitting
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled,train_target)

from sklearn.metrics import classification_report
predicted_target = lr.predict(test_scaled)
print(classification_report(test_target, predicted_target))


