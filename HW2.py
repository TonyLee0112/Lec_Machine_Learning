import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_percent = 0.4  # 40% of Train data
valid_percent = 0.25  # 1/4 of Train data = 10 % of Total data

# Modify the data location for loading data
data = np.loadtxt('data.csv', unpack=True, delimiter=',', skiprows=0)
data = np.transpose(data)

# Write a code for acquiring unbiased data
# 데이터를 섞어서 unbiased data를 얻을 수 있도록 설정
np.random.shuffle(data)

#Obtaining Training data set
train_end = int(len(data) * train_percent)
valid_end = int(train_end * valid_percent)

train_set = data[:train_end]
train_set = sorted(train_set, key=lambda train_set: train_set[0])  # Sorting again for data in order
train_set = np.transpose(train_set)

#Reallocate for efficient programming
train_x = train_set[0]  # train_set[0]: feature data set (i.e., x)
train_y = train_set[1]  # train_set[1]: label data set (i.e., y)

# Write code for obtaining valid data set: valid_set
valid_set = data[train_end:train_end + valid_end]
valid_set = np.transpose(valid_set)
valid_x = valid_set[0]
valid_y = valid_set[1]

# Write code for obtaining test data set: test_set
test_set = data[train_end + valid_end:]
test_set = np.transpose(test_set)
test_x = test_set[0]
test_y = test_set[1]

##################### Regression Libraries #############
def fit_polynomial(x, y, degree):
    '''
    Fits a polynomial to the input sample.
    (x, y): input sample
    degree: polynomial degree
    '''
    model = LinearRegression()
    model.fit(np.vander(x, degree + 1), y)
    return model

def apply_polynomial(model, x):
    '''
    Evaluates a linear regression model on an input sample
    model: linear regression model
    x: input sample
    '''
    degree = model.coef_.size - 1
    y = model.predict(np.vander(x, degree + 1))
    return y
##################### End of Regression Libraries #############

# Starting values
Optimal_Order = 0
Minimum_MSE = 9999
Optimal_Model = None

# 1. Polynomial의 order를 1차에서 10차까지 증가시며 각 order에서의 MSE performance을 
#    계산 해보고, MSE가 가장 낮은 최적의 polynomial order을 결정하라.
for polynomial_order in range(1, 11):
    model = fit_polynomial(train_x, train_y, polynomial_order)
    Estimated_valid_y = apply_polynomial(model, valid_x)

    # Write codes measuring MSE for valid set
    MSE = mean_squared_error(valid_y, Estimated_valid_y)

    if Minimum_MSE > MSE:
        Optimal_Order = polynomial_order
        Minimum_MSE = MSE
        Optimal_Model = model

print("----------------------\n")
print("We can choose best polynomial order with MSE of validation set.")
print(f"Optimal order is {Optimal_Order}")
print(f"Minimum MSE is {Minimum_MSE}\n")
print("-----TEST RESULT-----")

# 2. 최적의 Polynomial order이 구해졌으면, 이를 이용하여 Train set을 초록색 실선으로(g) Plot 
#    해보고, 그 그래프 위에 Test data set을 파란색 점으로(b.) Overlay시켜보아라.
plt.plot(train_x, apply_polynomial(Optimal_Model, train_x), 'g')   # Display with lines colored with green (g).
plt.plot(test_x, test_y, 'b.')  # Display with dots colored with blue (b).

# 3. 최종적으로 구해진 Optimal model과 Test data set을 이용해 MSE Performance를 출력하라. 
# Write code for calculating MSE performance of the Optimal regression polynomial.
test_pred_y = apply_polynomial(Optimal_Model, test_x)
MSE_Performance = mean_squared_error(test_y, test_pred_y)
print(f"MSE: {MSE_Performance}")

print(f"The coefficient of model is {Optimal_Model.coef_},  {Optimal_Model.intercept_}")

plt.xlabel('Feature values: x')
plt.ylabel('Label values: y')
plt.grid()
plt.suptitle('Polynomial Regression', fontsize=16)
plt.show()
