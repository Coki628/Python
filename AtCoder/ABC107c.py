# -*- coding: utf-8 -*-

N,K = map(int, input().split())
xN = list(map(int, input().split()))

ans = float('inf')
for i in range(N-K+1):
    tmp = float('inf')
    # 範囲が0を含む
    if xN[i] <= 0 <= xN[i+K-1]:
        # 距離の短い方を往復する
        if abs(xN[i]) <= abs(xN[i+K-1]):
            tmp = abs(xN[i])*2 + abs(xN[i+K-1])
        else:
            tmp = abs(xN[i]) + abs(xN[i+K-1])*2
    # 範囲が全て0より大きい
    elif xN[i] > 0:
        tmp = abs(xN[i+K-1])
    # 範囲が全て0より小さい
    elif xN[i+K-1] < 0:
        tmp = abs(xN[i])
    ans = min(tmp, ans)

print(ans)