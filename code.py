#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:
  def __init__(self):
    self.data_train = []
    self.data_test = []
    self.Label = []
  def Read_File(self, data_train, data_test):
    # Train set 과 Test set 데이터를 받아서 data_train 과 data_test 에 저장.
    path_train = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_train.csv'
    f_train = open(path_train, 'r', encoding='utf-8')
    for line in csv.reader(f_train):
      data_train.append(line)
    f_train.close()

    path_test = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_test.csv'
    f_test = open(path_test, 'r', encoding='utf-8')
    for line in csv.reader(f_test):
      data_test.append(line)
    f_test.close()
  def Write_File(self, data_test, label):
    path_write = 'C:/Users/leesooho/Desktop/Machine Learning/ML_HW#3_Knn/iris_test.csv'
    f_write = open(path_write, 'w', encoding='utf-8') 
    wr = csv.writer(f_write)
    for i in range(len(data_test)):
      data_test[i].append(label[i])
      wr.writerow([data_test[i][0], data_test[i][1], data_test[i][2], data_test[i][3], data_test[i][4]])
      print("[", i, "] ", data_test[i])

    f_write.close()
  # Complete Find_Neighbor
  def Find_Neighbor_Label(self, data_train, data_test, number_k):
    # data_train, data_test 모두 list 형식이고,값들이 정렬되어 있지 않다.
    # 정렬 기준을 만들어 정렬하거나 아니면 Neighbor 표본들을 수집할 모델을 선정.
    # KNeighborsClassifier 을 import
    # 모델 학습 -> 1. input 과 target 으로 분할 2. 학습
    # 120개의 데이터 idx가 0~3 : input, idx 가 4 : target
    def Find_Neighbor_Label(self, data_train, data_test, number_k):
      train_input = np.array([sample[:-1] for sample in data_train])
      train_target = np.array([sample[-1] for sample in data_train])
      data_test = np.array(data_test)  # 테스트 데이터는 한 번에 변환

      classifier = KNeighborsClassifier(n_neighbors=number_k)
      classifier.fit(train_input, train_target)

      self.Label = classifier.predict(data_test)


if __name__ == '__main__' :
  knn = KNN()

  knn.Read_File(knn.data_train, knn.data_test)

  knn_num =  int(input("Number of k : "))

  knn.Find_Neighbor_Label(knn.data_train, knn.data_test, knn_num)
  knn.Write_File(knn.data_test, knn.Label)
