# -*- coding: utf-8 -*-

N = int(input())
S = input()

# 累積和を使う方法
W_ruiseki = [0]
E_ruiseki = [0]
# 0地点を含めたN+1パターンを予め用意しておく
for i in range(N):
    if S[i] == "W":
        W_ruiseki.append(W_ruiseki[i] + 1)
    else:
        W_ruiseki.append(W_ruiseki[i])
    # Eは全体-Wで出せる
    E_ruiseki.append(i+1 - W_ruiseki[i+1])

ans = float("inf")
for i in range(N):
    cnt = 0
    cnt += W_ruiseki[i]
    cnt += E_ruiseki[N] - E_ruiseki[i+1]
    ans = min(ans, cnt)

print(ans)　