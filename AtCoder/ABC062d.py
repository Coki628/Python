# -*- coding: utf-8 -*-

"""
・TLE
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from bisect import insort, bisect_left

N = int(input())
aN = list(map(int, input().split()))
mids = aN[N:N*2]
L = sorted(aN[:N])
R = sorted(aN[N:])

ans = sum(L) - sum(R[:N])
for i in range(N):
    insort(L, mids[i])
    R.pop(bisect_left(R, mids[i]))
    ans = max(sum(L[-N:]) - sum(R[:N]), ans)
print(ans)
