"""
・自力AC！
・再帰探索
・値はすぐでかくなると判断して、再帰で行った。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def rec(sm, cnt, h, x):
    if sm > x:
        return 0
    return rec(sm+cnt*2+(h+1)**2, cnt*2+(h+1)**2, h*2+1, x) + 1

for _ in range(INT()):
    x = INT()
    ans = rec(1, 1, 1, x)
    print(ans)
