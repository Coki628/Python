# -*- coding: utf-8 -*-

N = int(input())
bN = list(map(int, input().split()))

aN = [0] * N
aN[0] = 0
aN[-1] = bN[0]
for i in range(1, N // 2):
    # 今回のbを、前回の後ろのaと比べる
    if bN[i] <= aN[N-i]:
        aN[i] = 0
        aN[N-i-1] = bN[i]
    else:
        aN[i] = bN[i] - aN[N-i]
        aN[N-i-1] = aN[N-i]
print(*aN)
