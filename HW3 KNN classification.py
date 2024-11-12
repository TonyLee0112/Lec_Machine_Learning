import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:
    def __init__(self):
        self.data_train = []
        self.data_test = []
        self.Label = []

    def Read_File(self, data_train, data_test):
        # Train 데이터 읽기
        path_train = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_train.csv'
        with open(path_train, 'r', encoding='utf-8') as f_train:
            for line in csv.reader(f_train):
                if line:  # 빈 줄 방지
                    data_train.append([float(x) for x in line[:-1]])  # 특징 데이터 저장
                    self.Label.append(float(line[-1]))  # 라벨 데이터 저장

        # Test 데이터 읽기
        path_test = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_test.csv'
        with open(path_test, 'r', encoding='utf-8') as f_test:
            for line in csv.reader(f_test):
                if line:  # 빈 줄 방지
                    data_test.append([float(x) for x in line])  # 테스트 데이터 저장

    def Write_File(self, data_test, label):
        path_write = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_test.csv'
        f_write = open(path_write, 'w', encoding='utf-8',newline='')
        if f_write:
            wr = csv.writer(f_write)
            for i in range(len(data_test)):
                data_test[i].append(label[i])
                wr.writerow([data_test[i][0], data_test[i][1], data_test[i][2], data_test[i][3], label[i]])
                print("[", i, "] ", data_test[i])
        else :
            print("Write File Error")

        f_write.close()

    # Complete Find_Neighbor
    def Find_Neighbor_Label(self, data_train, data_test, number_k):
        # 데이터 분리 및 변환
        train_input = np.array([sample[:len(sample)] for sample in data_train], dtype=float)
        train_target = np.array([int(sample[-1]) for sample in data_train], dtype=int)

        # data_test 비어 있는지 확인
        if not data_test:
            print("Error: data_test is empty. Please check the input data.")
            return
        test_set = np.array(data_test)[:, :4]
        classifier = KNeighborsClassifier(n_neighbors=number_k)
        classifier.fit(train_input, train_target)
        pred_list = classifier.predict(test_set)
        self.Label = [10 if p == 0 else 20 if p == 1 else 30 for p in pred_list]
        
if __name__ == '__main__':
    knn = KNN()
    knn.Read_File(knn.data_train, knn.data_test)
    knn_num = int(input("Number of k : "))
    knn.Find_Neighbor_Label(knn.data_train, knn.data_test, knn_num)
    knn.Write_File(knn.data_test, knn.Label)
