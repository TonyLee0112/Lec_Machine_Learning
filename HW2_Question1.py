# Polynomial Regression
# Train_set : 40 %, Test_set : 60 %
# Verification_set : 10 % of Train_set

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load data
# data.shape = (1999,2)
# x = feature, y = target
df = pd.read_csv("data.csv")
# df 을 input nparray 와 target nparray로 변환
# df[0] 은 indexing 이 아님. 열 이름이 0인 것을 반환
# df.iloc[x1:x2, y1:y2]
input_data = df.iloc[:,0].values.reshape(-1,1) # -1 : 알아서 차원 맞춰라
target_data = df.iloc[:,1].values

# Data set 분할 비율 설정
Test_ratio = 0.6 # Total_set 의 0.6배 = Test_set
Valid_ratio = 0.1 # Train_set 의 0.1배 = Valid_set

train_input, test_input, train_target, test_target = train_test_split(input_data,target_data,test_size = Test_ratio)
sub_input, val_input, sub_target, val_target = train_test_split(train_input,train_target,test_size = Valid_ratio)


# 1. Polynomial 의 order 를 1~10차까지 증가시키며 각 order 에서의 MSE performance 를 계산,
# MSE 가 가장 낮은 최적의 order 를 찾아라.

MSE_scores = []
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
for order in range(1,11) :
    poly = PolynomialFeatures(degree=order)
    poly_input = poly.fit_transform(sub_input) # 0차부터 ~ order 차 까지 데이터 생성

    # model 학습
    lr = LinearRegression()
    lr.fit(poly_input,sub_target)

    # valid_data 변환
    poly_valid = poly.transform(val_input)
    val_predict = lr.predict(poly_valid)

    # 최소 제곱 오차 계산 : sum((정답 - 예측)^2)
    mse = mean_squared_error(val_target,val_predict)
    MSE_scores.append(mse)

opt_order = np.argmin(MSE_scores) + 1
print(f"최적의 차수는 : {opt_order}")
