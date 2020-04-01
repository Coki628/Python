# -*- coding: utf-8 -*-

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
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

for _ in range(INT()):
    N, M = MAP()
    if N < M:
        print(0)
        continue

    lastd = int(str(M)[-1])
    cnt10 = 0
    for i in range(1, 11):
        cnt10 += int(str(lastd*i)[-1])

    d, m = divmod(N, M*10)
    ans = d * cnt10
    for i in range(1, m//M+1):
        ans += int(str(lastd*i)[-1])
    print(ans)
