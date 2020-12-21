# -*- coding: utf-8 -*-

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

N, K = map(int, input().split())
aN = list(map(int, input().split()))

aN.sort(reverse=True)

need = [False] * N
for i in range(N):
    if aN[i] >= K:
        need[i] = True
    else:
        sm = aN[i]
        for j in range(i+1, N):
            sm += aN[j]
            # 複数で条件を満たせば、それに含まれるものは必要
            if sm >= K:
                for k in range(i, j+1):
                    need[k] = True
                break
print(need.count(False))
