# -*- coding: utf-8 -*-

"""
・WA
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

l = sum(L)
r = sum(R[:N])
ans = l - r
for i in range(N):
    insort(L, mids[i])
    l += mids[i]
    l -= L[i]
    if mids[i] >= R[N]:
        R.pop(bisect_left(R, mids[i]))
    else:
        R.pop(bisect_left(R, mids[i]))
        r -= mids[i]
        r += R[-(N-i)]
    ans = max(l - r, ans)
print(ans)
