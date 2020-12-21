# -*- coding: utf-8 -*-

"""
・自力TLE。。
"""

import sys
from collections import Counter

def input(): return sys.stdin.buffer.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, Q = MAP()
A = LIST()
MAX = 32

C = Counter(A)
out = []
for _ in range(Q):
    q = INT()
    tmp = C.copy()
    ans = 0
    for k in range(MAX-1, -1, -1):
        if (q>>k) & 1:
            need = 1 << k
            cnt = 0
            powk = 2**k
            while C[powk] * powk < need:
                need -= C[powk] * powk
                cnt += C[powk]
                C[powk] = 0
                k -= 1
                if k == -1:
                    break
                powk = 2**k
            if k == -1:
                out.append(str(-1))
                break
            else:
                cnt += need // powk
                C[powk] -= need // powk
                ans += cnt
    else:
        out.append(str(ans))
    C = tmp
print('\n'.join(out))
