# -*- coding: utf-8 -*-

import sys
from itertools import accumulate

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

N, M = MAP()
diff = [0] * N
sm = 0
for i in range(N):
    a, b = MAP()
    diff[i] = a - b
    sm += a

if sm <= M:
    print(0)
    exit()
diff.sort(reverse=1)
ans = 0
for d in diff:
    sm -= d
    ans += 1
    if sm <= M:
        print(ans)
        exit()
print(-1)
