# -*- coding: utf-8 -*-

N, Y = map(int, input().split())

# kは全体-(i+j)で出せるのでループは2重でいい
for i in range(N+1):
    for j in range(N+1):
        k = N - (i + j)
        # kがマイナスになったらその先は周回する必要なし
        if k < 0:
            break
        # 条件を満たす値なら出力して終了
        if i * 10000 + j * 5000 + k * 1000 == Y:
            print(i, j, k)
            exit()
# 1つも候補がなかった場合
print(-1, -1, -1)