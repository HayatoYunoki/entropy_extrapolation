def main():
    N = 1000
    dt = 0.1
    T = 10
    square_wave(N, dt, T)
    ##print("a")

def square_wave(N, dt, T):
    count = 0
    t = 0
    y = 1
    f = open('square.txt', 'w')
    w_text = ""
    for i in range(int(N/T)):
        for j in range(T):
            w_text += str(t) + ", " + str(y) + "\n"
            t += dt
            count += 1
            if count == 1000:
                break
        y *= -1
    f.write(w_text)
    f.close()


if __name__ == "__main__":
    main()