# -*- coding: utf-8 -*-

"""
・普通にfor文ループなのに何故かpypy間に合わなくてpythonでAC
・20万*6を1秒以内
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

from itertools import permutations

N = int(input())
S = input()

ans = INF
for perm in permutations('RGB'):
    cnt = 0
    t = ''
    for j in range(len(S)):
        t += perm[j%3]
        if S[j] != perm[j%3]:
            cnt += 1
    if cnt < ans:
        ans = cnt
        T = t
print(ans)
print(T)
