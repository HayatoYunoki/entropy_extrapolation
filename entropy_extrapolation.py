import numpy as np
import math
from scipy.optimize import minimize

def main():
    q = 4
    alpha = 0.25
    init_guess = 0.1
    file_path = "sin.txt"
    N = 4 #何個の点を予測するか
    pre_result = 0
    test_data = [0]*(2**q)
    for i in range(N):
        result = minimize(calc_entropy, init_guess, args = (alpha, q, file_path, test_data, i, pre_result))
        print(result.x)
        print(result.success)
        print(result.message)
        print(result.fun)
        pre_result = result.x[0]
        for i in range(len(test_data)-1):
            test_data[i] = test_data[i+1]

def make_tensor(q, test_data):
    tensor_shape = []
    for i in range(q):
        tensor_shape.append(2)
    c = np.full(2**q, 1.0).reshape(tensor_shape)
    for i in range(2**q):
        binary_string = bin(i)[2:].zfill(q) ##0bを除去
        binary_tuple = tuple(int(bit) for bit in binary_string)
        c[binary_tuple] = test_data[i]
    return c

def tensor_svd(c, q):
    c_matrix = c.reshape(2, -1)
    S = []
    for i in range(1, q):
        new_c_matrix = c_matrix.reshape(2**i, -1)
        A1, S12, A2 = np.linalg.svd(new_c_matrix)
        S.append(S12)
    return S

def read_data(init_guess, file_path, q, start_line, test_data, pre_result):
    if start_line == 0:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for i in range(start_line, min(2**q+start_line, len(lines))):
            test_data[i-start_line] = float(lines[i].strip())  # 行末の改行文字を取り除いてリストに追加
    test_data[-1] = init_guess[0]
    # print(test_data)
    return test_data

def calc_entropy(init_guess, alpha, q, file_path, test_data, start_line, pre_result):
    test_data = read_data(init_guess, file_path, q, start_line, test_data, pre_result)
    print(test_data)
    c = make_tensor(q, test_data)
    S = tensor_svd(c, q) #Sは特異値ベクトルを格納するリスト
    H = 0.0
    for j in range(q-1):
        #Sを規格化
        norm_S_2 = 0.0
        for i in range(2):
            norm_S_2 += S[j][i]**2
        sum = 0.0
        for i in range(2):
            sum += ((S[j][i]**2)/norm_S_2)**alpha
        H += math.log(sum)/(1-alpha)
    return H

if __name__ == "__main__":
    main()