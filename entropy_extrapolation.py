import numpy as np
import math
from scipy.optimize import minimize

def main():
    q = 8
    alpha = 0.4
    file_path = "square.txt"
    outfile_path = "square_out.txt"
    N = 150 #何個の点を予測するか
    pre_result = 0
    test_data = [0]*(2**q)
    f = open(outfile_path, 'w')
    out_text = ""
    ent_out_text = ""

    for i in range(N):
        result = minimize(calc_entropy, np.random.rand(), args = (alpha, q, file_path, test_data, i, pre_result), method = 'L-BFGS-B')

        if i == 193:#エンタングルメントエントロピーの振る舞いを調べる
            for k in range(100):
                ent_out_text += str(calc_entropy([-1.0+0.02*k], alpha, q, file_path, test_data, i, pre_result)) + "\n"

        print(result.x)
        print(result.success)
        if result.success == False:
            print("failed")
            # break
            result.x[0] = pre_result
        print(result.message)
        print(result.fun)
        pre_result = result.x[0]
        for i in range(len(test_data)-1):
            test_data[i] = test_data[i+1]
        out_text += str(result.x[0]) + "\n"

    f.write(out_text)
    f.close()


    ent_f = open("ent_out.txt", 'w')
    ent_f.write(ent_out_text)
    ent_f.close()
    

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
        A1, S12, A2 = np.linalg.svd(c_matrix)
        print("A1:{}".format(A1))
        print("S12:{}".format(S12))
        print("A2:{}".format(A2))
        num_row, num_column = c_matrix.shape
        if num_row < num_column:
            c_matrix = np.dot(np.diag(S12), A2[:2**i, :])
            c_matrix = c_matrix.reshape(2**(i+1), -1)
        else:
            c_matrix = np.dot(np.diag(S12), A2)
        print("c_matrix:{}".format(c_matrix))
        S.append(S12)
    return S

def read_data(init_guess, file_path, q, start_line, test_data, pre_result):
    if start_line == 0:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for i in range(start_line, min(2**q+start_line, len(lines))):
            test_data[i-start_line] = float(lines[i].strip())  # 行末の改行文字を取り除いてリストに追加
    test_data[-1] = init_guess[0]
    print(test_data)
    return test_data

def make_s_dim(q): #サイトごとの特異値の個数を格納したリストを返す
    s_dim_list = []
    if q % 2 == 0:
        midpoint = q // 2
    else:
        midpoint = (q // 2) + 1

    for i in range(1, midpoint + 1):
        s_dim_list.append(2**i)

    for i in range(midpoint, 0, -1):
        s_dim_list.append(2**i)
    print("s_dim_list: {}".format(s_dim_list))
    return s_dim_list

def calc_entropy(init_guess, alpha, q, file_path, test_data, start_line, pre_result):
    test_data = read_data(init_guess, file_path, q, start_line, test_data, pre_result)
    # print(test_data)
    c = make_tensor(q, test_data)
    # s_dim_list = make_s_dim(q-1)
    S = tensor_svd(c, q) #Sは特異値ベクトルを格納するリスト
    H = 0.0
    for j in range(q-1):#jはサイト
        #Sを規格化
        norm_S_2 = 0.0
        for i in range(len(S[j])):
            norm_S_2 += S[j][i]**2
        sum = 0.0
        for i in range(len(S[j])):
            sum += ((S[j][i]**2)/norm_S_2)**alpha
        H += math.log(sum)/(1-alpha)
    return H

if __name__ == "__main__":
    main()