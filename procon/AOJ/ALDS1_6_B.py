# -*- coding: utf-8 -*-

"""
パーティション
"""

N = int(input())
aN = list(map(int, input().split()))

def partition(p, r):
    x = aN[r]
    i = p-1
    for j in range(p, r):
        if aN[j] <= x:
            i = i+1
            aN[i], aN[j] = aN[j], aN[i]
    aN[i+1], aN[r] = aN[r], aN[i+1]
    # 基準となった末尾要素のindex
    return i+1

idx = partition(0, N-1)
aN[idx] = '[' + str(aN[idx]) + ']'
print(*aN)
