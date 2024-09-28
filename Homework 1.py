import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv') # 149 송이의 아이리스 꽃의 5 Features 데이터
Iris_data = df.to_numpy()

Feature = ['Sepal_length','Sepal_width','Petal_Length','Petal_Width','Class']
# 아이리스의 품종은 10,20,30 인 숫자로 mapping 되어있다.
# 1. 2개의 특성을 골라서 2D Plotting
i = 2; j = 3
plt.title("2D Plot")
plt.scatter(Iris_data[:,i],Iris_data[:,j])
plt.xlabel(Feature[i])
plt.ylabel(Feature[j])
plt.show()

# 2-1. 각 특성 조합에 대한 Correlation 값 계산 및 저장
Correlations = dict()
for a in range(4) :
    for b in range(4) :
        if a < b :
            x = Iris_data[:,a]
            y = Iris_data[:,b]
            Correlations[(Feature[a],Feature[b])] = np.corrcoef(x, y)[0, 1]

for key,value in Correlations.items() :
    print(key,f"{value:.2f}")

# Correlation 크기 순서
# (2,3) > (0,1) > (0,2) > 0.5 > (1,2) > (1,3) > (0,1)

# 2-2. 상관계수의 크기가 가장 작은 쌍 찾기
min_abs_key = min(Correlations, key=lambda k: abs(Correlations[k]))
f1,f2 = min_abs_key
i = Feature.index(f1)
j = Feature.index(f2)

# # 2D Plotting
plt.title("Chosen Features")
plt.scatter(Iris_data[:,i],Iris_data[:,j])
plt.xlabel(Feature[i])
plt.ylabel(Feature[j])
plt.show()

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 2; j = 3
X = Iris_data[:,i]
Y = Iris_data[:,j]
Z = Iris_data[:,-1]
ax.scatter(X,Y,Z,c=Iris_data[:,-1],cmap='viridis',edgecolor='k')
ax.set_xlabel(Feature[i])
ax.set_ylabel(Feature[j])
ax.set_zlabel(Feature[-1])
plt.suptitle('3D Plotting', fontsize = 16)
plt.show()

# 특징 분석 결과, class = 10 인 꽃들은 잘 구분이 되나, class = 20 과 30은 구분이 잘 안됨.
# 20 과 30을 구분할 수 있는 새로운 parameter 을 만들어야함.
# 아이러니 하게도 Correlation 크기가 가장 큰 Petal_Length 와 Petal_Width 조합이 가장 품종 구분을 잘함.
# Petal_width 는 데이터의 범위가 0~2.5 인 반면에, Petal Length는 0~7 이라 반경이 더 넓음.
# 넓은 애를 더 키울까 아니면 좁은 애를 더 키울까?
# Data Preprocessing vs Feature Engineering?
# Data Preprocessing 은 SVM, Logistic Regression, K-Nearest Neighbor, 신경망에서 유리
# Feature Engineering 은 Classification, 특성쌍의 비종속성을 반영하는데 유리

# 4. Feature 합성
i = 2; j = 3
New_Feature = Iris_data[:,i] / Iris_data[:,j]
New_Iris_data = np.insert(Iris_data,-1,New_Feature,axis=1)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
X = New_Iris_data[:,i]
Y = New_Iris_data[:,j]
Z = New_Iris_data[:,-2]
ax.scatter(X,Y,Z,c=Iris_data[:,-1],cmap='viridis',edgecolor='k')
ax.set_xlabel(Feature[i])
ax.set_ylabel(Feature[j])
ax.set_zlabel('New Feature')
plt.suptitle('3D Plotting', fontsize = 16)
plt.show()
