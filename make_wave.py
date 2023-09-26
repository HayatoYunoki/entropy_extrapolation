import math

def main():
    N = 1000
    dt = 0.1
    T = 10
    square_wave(N, dt, T)
    sin_wave(N, dt)
    triangle_wave(N, dt, T)
    #print("a")

def square_wave(N, dt, T):
    count = 0
    t = 0
    y = 1
    f = open('square.txt', 'w')
    w_text = ""
    for i in range(int(N/T)):
        for j in range(T):
            w_text += str(y) + "\n"
            t += dt
            count += 1
            if count == N:
                break
        y *= -1
    f.write(w_text)
    f.close()

def sin_wave(N, dt):
    t = 0
    f = open('sin.txt', 'w')
    w_text = ""
    for i in range(N):
        w_text += str(math.sin(t)) + "\n"
        t += dt
    f.write(w_text)
    f.close()

def triangle_wave(N, dt, T):
    count = 0
    t = 0
    f = open('triangle.txt', 'w')
    y = -0.2
    dy = 0.04
    w_text = ""
    for i in range(int(N/T)):
        for j in range(T):
            y += dy
            w_text += str(y) + "\n"
            count += 1
            if count == N:
                break
        dy *= -1
    f.write(w_text)
    f.close()


if __name__ == "__main__":
    main()