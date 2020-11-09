"""
・自力AC！
・累積和と、累積和の累積maxを取る。
　いわゆる添字バグりがちな系だけど、一発で通ってよかった。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = LIST()

acc = list(accumulate([0]+A))
accmx = list(accumulate(acc, max))

ans = 0
cur = 0
for i in range(1, N+1):
    ans = max(ans, cur+accmx[i])
    cur += acc[i]
print(ans)
