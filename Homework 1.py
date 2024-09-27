import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv') # 149 개의 꽃 데이터. 각 5개씩
Iris_data = df.to_numpy()

# 1. 4 Features 중 2개씩 선택하여 2D plot
Feature_names = ['Sepal_length','Sepal_width','Petal_Length','Petal_Width','Class']
# 1-1 Sepal Length, Sepal Width

i = 2; j = 3

plt.scatter(Iris_data[:,i],Iris_data[:,j])
plt.xlabel(Feature_names[i])
plt.ylabel(Feature_names[j])
# plt.show()

# Correlation 계산
Correlations = dict()
for a in range(4) :
    for b in range(4) :
        if a < b :
            x = Iris_data[:,a]
            y = Iris_data[:,b]
            Correlations[(Feature_names[a],Feature_names[b])] = np.corrcoef(x, y)[0, 1]

for key,value in Correlations.items() :
    print(key,f"{value:.2f}") # 실제 값은 변하지 않고 print 형식만 조정

