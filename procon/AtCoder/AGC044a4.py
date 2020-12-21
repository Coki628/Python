"""
参考：誰かのツイートとか
・メモ化再帰
・やっぱlog乗らない分こっちのが全然速い。pypyAC0.4秒。
"""

import sys
from collections import defaultdict

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7

def rec(u):
    if dist[u] != INF:
        return dist[u]
    if u == 0:
        return 0
    res = abs(u)*d
    if u >= 5:
        x = u % 5
        v = (u-x) // 5
        res = min(res, rec(v) + c+d*x)
        if u % 5 != 0:
            x = 5 - u % 5
            v = (u+x) // 5
            res = min(res, rec(v) + c+d*x)
    if u >= 3:
        x = u % 3
        v = (u-x) // 3
        res = min(res, rec(v) + b+d*x)
        if u % 3 != 0:
            x = 3 - u % 3
            v = (u+x) // 3
            res = min(res, rec(v) + b+d*x)
    if u >= 2:
        x = u % 2
        v = (u-x) // 2
        res = min(res, rec(v) + a+d*x)
        if u % 2 != 0:
            x = 2 - u % 2
            v = (u+x) // 2
            res = min(res, rec(v) + a+d*x)
    dist[u] = res
    return res

for _ in range(INT()):
    N, a, b, c, d = MAP()

    dist = defaultdict(lambda: INF)
    ans = rec(N)
    print(ans)
