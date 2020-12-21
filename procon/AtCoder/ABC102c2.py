# -*- coding: utf-8 -*-

N = int(input())
AN = list(map(int, input().split()))

# Ai - i の各値を用意してソートする
BN = [AN[i] - (i + 1) for i in range(N)]
BN.sort()
# 中央値を取る
b = BN[len(BN) // 2]
# 最適な b で実際の計算をする
ans = 0
for i in range(N):
    ans += abs(AN[i] - (b + i + 1))
print(ans)