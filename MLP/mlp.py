#2011105073 이영석
#Multi Layer Perceptron

import random
import time
import numpy as np

hidden_node_num = 300
eta = 0.1
#w 값 초기 랜덤 생성
w = [0, 1]
w[0] = np.zeros((64, hidden_node_num))
for i in range(64):
    for j in range(hidden_node_num):
        w[0][i][j] = random.uniform(0.001, 0.009)
w[1] = np.zeros((hidden_node_num, 10))
for i in range(hidden_node_num):
    for j in range(10):
        w[1][i][j] = random.uniform(0.001, 0.009)
#target value 설정
t = []
for i in range(10):
    t.append([])
    temp = []
    for j in range(10):
        if j == i:
            temp.append(1)
        else:
            temp.append(0)
    t[i] = temp

#sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

#file_name 이름을 가진 파일을 읽어서 int형 리스트로 return
def read_file(file_name):
    f = open(file_name, 'r')
    lines = f.read().splitlines()
    f.close()
    data_lines = []
    for line in lines:
        split_line = line.split(',')
        data_lines.append(list(map(int, split_line)))
    return data_lines

# 학습 완료 체크
# 예시: 0인 경우 첫자리 0.9 이상 나머지자리 0.01 이하로 나와야 학습 완료
def test_learning(number, o):
    max = 0
    max_num = -1
    for i in range(10):
        if o[i] > max:
            max = o[i]
            max_num = i
    if number == max_num:
        return True
    else:
        return False

#error rate 계산
def calculate_error(number, o):
    error = 0
    for i in range(10):
        error += pow(t[number][i] - o[i], 2)
    return error / 2

#trail 데이터로 학습
def learning():
    data_lines = read_file("optdigits.tra")
    cnt = 0
    pass_error_rate = 0
    while True:
        for line in data_lines:
            number = line[64]
            x = np.array(line[0:64], dtype=float)
            o1 = sigmoid(np.dot(x, w[0]))
            o2 = sigmoid(np.dot(o1, w[1]))
            #print(cnt, number, o2)
            #error rate 계산후 종료
            if calculate_error(number, o2) < 0.0001:
                pass_error_rate += 1
            else:
                pass_error_rate = 0
            if pass_error_rate > 20:
                return cnt
            # delta w 계산
            l2_delta = (t[number] - o2) * o2 * (1 - o2)
            l1_delta = o1 * (1 - o1) * np.dot(w[1], l2_delta)
            w[1] += np.outer(o1, l2_delta) * eta
            w[0] += np.outer(x, l1_delta) * eta
            cnt += 1

#test 데이터로 학습 결과 분석
def test(cnt):
    test_lines = read_file("optdigits.tes")
    correct = 0
    bad = 0
    for line in test_lines:
        number = line[64]
        x = np.array(line[0:64], dtype=float)
        o1 = sigmoid(np.dot(x, w[0]))
        o2 = sigmoid(np.dot(o1, w[1]))
        if test_learning(number, o2):
            correct += 1
        else:
            bad += 1
    print("%d번 트레일 데이터 학습을 하여 테스트 데이터에 %0.2f%% 일치합니다." % (cnt, correct / (correct + bad) * 100))


start_time = time.time()
cnt = learning()
end_time = time.time()
print("%0.0f분 %0.0f초 걸렸습니다." % ((end_time - start_time) / 60 , (end_time - start_time) % 60))
test(cnt)