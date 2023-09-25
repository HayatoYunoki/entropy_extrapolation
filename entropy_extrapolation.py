import numpy as np

def main():
    q = 3
    alpha = 0.25
    c = make_tensor(q)
    S = tensor_svd(c, q) #Sは特異値ベクトルを格納するリスト

def make_tensor(q):
    tensor_shape = []
    test_data = []
    for i in range(q):
        tensor_shape.append(2)
    for i in range(2**q):
        test_data.append(i+1)
    ##print(test_data)
    c = np.array(range(2**q)).reshape(tensor_shape)
    for i in range(2**q):
        binary_string = bin(i)[2:].zfill(q) ##0bを除去
        binary_tuple = tuple(int(bit) for bit in binary_string)
        # print(binary_tuple)
        c[binary_tuple] = test_data[i]
    # print(c[0, 1, 1])
    return c

def tensor_svd(c, q):
    c_matrix = c.reshape(2, -1)
    # U = np.array([[0.376168, -0.92655],[0.92655, 0.376168]])
    # sigma = np.array([[14.2274, 0],[0, 1.25733]])
    # Vt = np.array([[0.352062, 0.443626, 0.53519, 0.626754],[0.758981, 0.321242, -0.116498, -0.554238]])
    # print(U@sigma@Vt)
    S = []
    A1, S12, A2 = np.linalg.svd(c_matrix)
    S.append(S12)
    S12 = (S12 * np.eye(2))
    A2 = A2[:2]
    print(A1@S12@A2)
    # return S


if __name__ == "__main__":
    main()