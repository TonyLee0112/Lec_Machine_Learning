import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv') # 149 개의 꽃 데이터. 각 5개씩
Iris_data = df.to_numpy()

Feature_names = ['Sepal_length','Sepal_width','Petal_Length','Petal_Width','Class']
# 1. Making 2D plot with 2 Features among the given file
i = 2; j = 3
plt.title("2D Plot")
plt.scatter(Iris_data[:,i],Iris_data[:,j])
plt.xlabel(Feature_names[i])
plt.ylabel(Feature_names[j])
plt.show()

# 2-1. Correlation List of Each Feature pair
Correlations = dict()
for a in range(4) :
    for b in range(4) :
        if a < b :
            x = Iris_data[:,a]
            y = Iris_data[:,b]
            Correlations[(Feature_names[a],Feature_names[b])] = np.corrcoef(x, y)[0, 1]

for key,value in Correlations.items() :
    print(key,f"{value:.2f}") # 실제 값은 변하지 않고 print 형식만 조정

# 2-2. Choose one optimal Feature pair
# 상관계수의 크기가 가장 작은 ('Sepal_length', 'Sepal_width') 쌍
plt.title("Chosen Features")
plt.scatter(Iris_data[:,0],Iris_data[:,1])
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')
plt.show()

# 3. 3D Plot with chosen Features
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

X = Iris_data[:,0]
Y = Iris_data[:,1]
Z = Iris_data[:,4]
ax.scatter(X,Y,Z)
ax.set_xlabel('Sepal_length')
ax.set_ylabel('Sepal_width')
ax.set_zlabel('class')
plt.suptitle('3D Plotting', fontsize = 16)
plt.show()
