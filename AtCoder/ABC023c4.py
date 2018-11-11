# -*- coding: utf-8 -*-

"""
正答より。
データの持ち方をくるくる変えて答えを導くのが
自分だとうまく思いつかなかった…。
"""

R,C,K = map(int, input().split())
N = int(input())

# アメの位置情報
rN = [0] * N
cN = [0] * N
# 行Rと列Cのアメ合計数
r_sum = [0] * R
c_sum = [0] * C
for i in range(N):
    rN[i],cN[i] = map(int, input().split())
    # 添字調整
    rN[i] -= 1
    cN[i] -= 1
    r_sum[rN[i]] += 1
    c_sum[cN[i]] += 1

# アメの個数に対する行列数
x = [0] * (N+1)
y = [0] * (N+1)
for i in range(R):
    x[r_sum[i]] += 1
for i in range(C):
    y[c_sum[i]] += 1

# 各行列からのアメ合計がKになる行列の組み合わせ数
ans = 0
for i in range(K+1):
    ans += x[i] * y[K-i]

# 重複対策
for i in range(N):
    if r_sum[rN[i]] + c_sum[cN[i]] == K:
        ans -= 1
    if r_sum[rN[i]] + c_sum[cN[i]] == K+1:
        ans += 1
    
print(ans)
