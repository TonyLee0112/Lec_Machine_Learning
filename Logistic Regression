# Logistic Regression
# Probability of Lucky Bag
# Data Preprocessing

# 1. Pandas 에서 Data 읽어오기
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish.head()

# 2. 주어진 2D Array 를 Numpy Array 로 변환하기
fish_input = fish[['Weight','Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy() # fish 에서 'Species' 에 해당되는 열만 Numpy Array 로 변환

# 3. Train set 과 Test set 분할하기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input,fish_target, random_state=42)

# 4. Input Data Pre-processing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input) # StandardScaler only requires one parameter
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 5-1. K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba,decimals=4))
distances, indexes = kn.kneighbors(test_scaled[3:4])
# print(train_target[indexes])

# 5-2-1. Logistic Regression - Binary Classification
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt') # True, False 로 구성된 1D Array
train_bream_smelt = train_scaled[bream_smelt_indexes] # bream_smelt_indexes 의 True 에 해당하는 원소들의 Properties 들을 원소별로 반환
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

decisions = lr.decision_function(train_bream_smelt[:5]) # z value
from scipy.special import expit
# print(expit(decisions))

# 5-2-2 Logistic Regression - Multiple Classification
lr = LogisticRegression(C=20,max_iter=1000)
lr.fit(train_scaled,train_target)
proba = lr.predict_proba(test_scaled[:5])
decision = lr.decision_function(test_scaled[:5])
from scipy.special import softmax
proba = softmax(decision,axis=1)
print(np.round(proba,decimals=3))
