# -*- coding: utf-8 -*-

# 階乗を使う
import math

N, M = map(int, input().split())

ans = 0
if abs(N -M) >= 2:
    print(0)
    exit()
elif abs(N - M) == 1:
    ans = math.factorial(N) * math.factorial(M)
elif abs(N - M) == 0:
    ans = math.factorial(N) * math.factorial(M) * 2

print(ans % (10 ** 9 + 7))