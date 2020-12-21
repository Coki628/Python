# -*- coding: utf-8 -*-

"""
計算で解く(解説参考)
"""

N, H = map(int, input().split())
A, B, C, D, E = map(int, input().split())

ans = float('inf')
for i in range(N+1):
    for j in range(N+1):
        if H + B*i + D*j - (N - i - j) * E > 0:
            ans = min(A*i + C*j, ans)
print(ans)
