# -*- coding: utf-8 -*-

from bisect import bisect_right

N, X, Y = map(int, input().split())
aN = list(map(int, input().split()))
aN.sort()
weaks = bisect_right(aN, X)

# 同率or守りのが強い
if X <= Y:
    # 一撃で壊せるドアの半分切り上げ
    print(-(-weaks // 2))
# 攻めのが強い
else:
    # 実際にやってみる
    rest = 500
    for i in range(1000):
        if i % 2 == 0:
            rest -= 1
            for j in range(N):
                if aN[j] > 0:
                    aN[j] -= X
                    break
            if aN[-1] <= 0:
                print(len(aN))
                exit()
        else:
            for j in range(N-1, -1, -1):
                if aN[j] <= X*rest:
                    aN[j] += Y
                    break
    score = 0
    for i in range(N):
        if aN[i] <= 0:
            score += 1
    print(score)
