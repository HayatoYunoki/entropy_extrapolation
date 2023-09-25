import numpy as np
import math
from scipy.optimize import minimize

def main():
    q = 5
    alpha = 0.25
    init_guess = 25.0
    result = minimize(calc_entropy, init_guess, args = (alpha, q))
    print(result.x)
    print(result.fun)

def make_tensor(q, init_guess):
    tensor_shape = []
    test_data = []
    for i in range(q):
        tensor_shape.append(2)
    for i in range(2**q):
        test_data.append(i+1)
    test_data[2**q-1] = init_guess
    c = np.full(2**q, 1.0).reshape(tensor_shape)
    for i in range(2**q):
        binary_string = bin(i)[2:].zfill(q) ##0bを除去
        binary_tuple = tuple(int(bit) for bit in binary_string)
        c[binary_tuple] = test_data[i]
    return c

def tensor_svd(c, q):
    c_matrix = c.reshape(2, -1)
    # U = np.array([[0.376168, -0.92655],[0.92655, 0.376168]])
    # sigma = np.array([[14.2274, 0],[0, 1.25733]])
    # Vt = np.array([[0.352062, 0.443626, 0.53519, 0.626754],[0.758981, 0.321242, -0.116498, -0.554238]])
    # print(U@sigma@Vt)
    S = []
    for i in range(1, q):
        new_c_matrix = c_matrix.reshape(2**i, -1)
        # print(new_c_matrix)
        A1, S12, A2 = np.linalg.svd(new_c_matrix)
        S.append(S12)
        # S12 = (S12 * np.eye(2))
        # A2 = A2[:2]
        # c_matrix = A2
        # print(A1@S12@A2)
    return S

def calc_entropy(init_guess, alpha, q):
    c = make_tensor(q, init_guess)
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