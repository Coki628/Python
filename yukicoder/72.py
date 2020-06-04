"""
・自力AC？
・三分探索
・今度は制約が大きいので全探索は無理。
・前の問題の解法説明でヒントを得てしまっていたが、多分自分でも気づけたはず、多分。。
・てか両側に均等に割り振るのが最善。三分探索はしなくてもいい。こっちに気付いたかもしれない。
"""

import sys

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
MOD = 10 ** 6 + 7
EPS = 10 ** -10

def trisearch_max(lo, hi, func):
    """ 三分探索 """

    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 >= res2:
            hi = m2
        else:
            lo = m1
    return m1, m2

def calc(i):
    j = N - i
    return (i+1)*(j+1)-1

N = INT()
res1, res2 = trisearch_max(0, N, calc)
ans = max(calc(res1), calc(res2)) % MOD
print(ans)
