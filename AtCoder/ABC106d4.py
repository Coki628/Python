# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2018/08/18/224408
・二次元累積和2
・こっちの方針のが分かりやすいかも。
"""

N, M, Q = map(int, input().split())
LR = [[0] * (N) for i in range(N)]
for i in range(M):
    l, r = map(int, input().split())
    # 左端=L、右端=Rの表
    LR[l-1][r-1] += 1

# 二次元の縦横両方について累積和を出す
# 縦方向の足し方を逆からにすることで、今回の条件と合致できる
for i in range(N):
    for j in range(1, N):
        # 左端=L、右端<=Rになる
        LR[i][j] += LR[i][j-1]
for i in range(N):
    for j in range(N-2, -1, -1):
        # 左端>=L、右端<=Rになる
        LR[j][i] += LR[j+1][i]

for i in range(Q):
    p, q = map(int, input().split())
    # ここでは添字指定だけで加減不要
    ans = LR[p-1][q-1]
    print(ans)
