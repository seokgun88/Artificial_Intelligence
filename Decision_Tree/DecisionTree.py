#2011105073 이영석
#assignment1: decision tree

import numpy as np
import re
import queue

#decision tree의 노드 클래스
class Node:
    def __init__(self):
        self.attribute_num = 0
        self.next = []

#특정 attribute 관련 data 추출
def get_specific_data(data_lines, attribute_num, state):
    specific_data = []
    for line in data_lines:
        if line[attribute_num] == state:
            specific_data.append(line)
    return specific_data

#data에서 plus나 minus 추출
def get_plus_minus(data):
    plus = 0
    minus = 0
    for line in data:
        if line[0] == 'e':
            plus += 1
        else:
            minus += 1
    return plus, minus

#entropy 계산
def entropy(data, attribute_num, attribute=''):
    plus = 0
    minus = 0
    #attribute_num이 0이면 총 엔트로피
    if attribute == '':
        for line in data:
            if line[0] == 'e':
                plus += 1
            else:
                minus += 1
    #attribute_num이 0이 아니면 특정 attriute의 엔트로피
    else:
        for line in data:
            if line[attribute_num] == attribute:
                if line[0] == 'e':
                    plus += 1
                else:
                    minus += 1
    total = plus + minus
    if plus != 0:
        plus_entropy = -(plus/total * np.log2(plus/total))
    else:
        plus_entropy = 0
    if minus != 0:
        minus_entropy = -(minus/total * np.log2(minus/total))
    else:
        minus_entropy = 0
    return plus_entropy + minus_entropy

#data 속에서 attribute_num의 엔트로피 계산
def gain(data, attribute_num):
    total_entropy = entropy(data, attribute_num)
    attribute_entropy = 0
    total = len(data)
    for attribute in attributes[attribute_num - 1]:
        attribute_len = len(get_specific_data(data, attribute_num, attribute))
        attribute_entropy += attribute_len/total * entropy(data, attribute_num, attribute)
    return total_entropy - attribute_entropy

#data를 통해 eatable, poison 구분
def is_eatable(line):
    temp = root
    while True:
        if temp.attribute_num < 0:
            if temp.attribute_num == -1:
                return 'e' #eatable
            elif temp.attribute_num == -2:
                return 'p' #posion
            else:
                return 'u' #unknown
            break
        state_num = attributes[temp.attribute_num - 1].index(line[temp.attribute_num])
        temp = temp.next[state_num]

#decision tree 출력
def print_tree(node, level):
    if node.attribute_num == -1:
        print("Eatable (%d)" % level)
    elif node.attribute_num == -2:
        print("Poison (%d)" % level)
    elif node.attribute_num == -3:
        print("Unknown (%d)" % level)
    else:
        print("%d번 Attribute (%d)" % (node.attribute_num, level))
    for child_node in node.next:
        print_tree(child_node, level + 1)

#--read names--#
f = open("agaricus-lepiota.names.txt", 'r')
lines = f.read().splitlines()
is_attribute = False
is_first_line = True
attribute_num_pat = re.compile("(\d)[.]")
state_pat = re.compile("=([a-z])")
attributes = []
states = []
for line in lines:
    if is_attribute:
        attr_n = attribute_num_pat.search(line)
        if attr_n != None and not is_first_line:
            attributes.append(states)
            states = []
        if is_first_line:
            is_first_line = False
        state = state_pat.findall(line)
        if len(state) == 0:
            attributes.append(states)
            break
        states += state
    if line.find("Attribute Information") != -1:
        is_attribute = True
f.close()

#--read data--#
f = open("agaricus-lepiota.data.txt", 'r')
lines = f.read().splitlines()
data_lines = []
for line in lines:
    words = line.split(',')
    if words[11] == '?':
        continue
    data_lines.append(words)
f.close()

data_len = len(data_lines)
training_data = data_lines[0:int(data_len*0.9)]
test_data = data_lines[int(data_len*0.9):len(data_lines)]

#q:노드 큐, q2:노드 데이터 큐, q3:이미 트리에 추가된 attributes
q = queue.Queue()
q2 = queue.Queue()
q3 = queue.Queue()
root = Node()
pre_attributes = []
q.put(root)
q2.put(training_data)
q3.put(pre_attributes)

#training data를 분석해서 decision tree 생성
while True:
    if q.empty() or q2.empty() or q3.empty():
        break
    cur_node = q.get()
    cur_data = q2.get()
    pre_attributes = q3.get()

    #gain()을 이용하여 가장 영향력 큰 attribute 찾음
    attribute_num = 0
    attribute_entropy = 0
    for i in range(len(attributes)):
        if i in pre_attributes:
            continue
        gain_entropy = gain(cur_data, i + 1)
        if attribute_entropy < gain_entropy:
            attribute_num = i + 1
            attribute_entropy = gain_entropy

    #현재 노드에 최대 영향력 attribute 등록후 각 state 별로 자식 노드 생성 후 큐에 풋
    cur_node.attribute_num = attribute_num
    for state in attributes[attribute_num - 1]:
        state_data = get_specific_data(cur_data, attribute_num, state)
        plus, minus = get_plus_minus(state_data)
        child_node = Node()
        #해당 state 데이터가 전혀 없거나 eatable/poison이 확정된 경우
        if plus == 0 or minus == 0:
            if plus == 0 and minus == 0:
                child_node.attribute_num = -3 #non data
            elif plus == 0:
                child_node.attribute_num = -2 #poison
            elif minus == 0:
                child_node.attribute_num = -1 #eatable
            cur_node.next.append(child_node)
        #해당 state에서 다른 attribute에 따라서 다시 경우가 나뉘는 경우
        else:
            cur_node.next.append(child_node)
            q.put(cur_node.next[len(cur_node.next)-1])
            q2.put(state_data)
            pre_attributes.append(attribute_num - 1)
            q3.put(pre_attributes)

#test data를 decision tree로 결과 예측 후 평가
right_num = 0
wrong_num = 0
print("---테스트 데이터 확인 시작---")
for line in test_data:
    if line[0] == is_eatable(line):
        right_num += 1
        print("맞음!")
    else:
        wrong_num += 1
        print("틀림!")
print("총 %d개의 테스트 데이터에서 %0.2f%% 일치합니다." % (right_num + wrong_num, right_num/(right_num + wrong_num) * 100))

#decision tree preorder로 출력
print("---preorder로 decision tree 출력---")
print("Attribute (Tree Level)")
print_tree(root, 1)