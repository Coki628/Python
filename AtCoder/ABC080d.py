# -*- coding: utf-8 -*-

"""
自力AC！
・いもす、累積和をちょっと応用
・O((C*10^5)*3)=900万が重かったのか、TLEしたからpypyでAC
"""

N, C = map(int, input().split())
T = 10**5+2

c_ls = [[0] * T for _ in range(C)]
# いもす
for i in range(N):
    s, t, c = map(int, input().split())
    c_ls[c-1][s] += 1
    c_ls[c-1][t] -= 1
# 各チャンネル毎の累積和
for i in range(C):
    for j in range(1, T):
        c_ls[i][j] += c_ls[i][j-1]
# 他チャンネルとの重複を見るために開始を1早める(入力は全部整数なので-0.5は-1扱いでOKのはず)
for i in range(C):
    for j in range(1, T):
        if c_ls[i][j] > c_ls[i][j-1]:
            c_ls[i][j-1] += 1
# 全チャンネルを合計
sm = [0] * T
for i in range(C):
    for j in range(T):
        sm[j] += c_ls[i][j]
# 重複が一番多くなる箇所が答え
print(max(sm))
