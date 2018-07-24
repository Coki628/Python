# -*- coding: utf-8 -*-

N = int(input())
AN = list(map(int, input().split()))

BN = [0] * (N//2+1)

# 偶奇で分けて集計
for i in range(N):
    if N % 2 != 0:
        BN[N//2 - AN[i]//2] += 1
    else:
        BN[N//2 - (AN[i]//2+1)] += 1

# 奇数の場合は真ん中を考慮
for i in range(len(BN)):
    if BN[i] > 2 or N % 2 != 0 and BN[-1] > 1:
        print(0)
        exit()

# 結果出力
print(2 ** (N//2) % (10 ** 9 +7))