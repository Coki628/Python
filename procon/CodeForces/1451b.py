"""
・結構さくっと自力AC
・ようは連続部分列から、左右どっちかの端1個だけをずらして同じものが作れればOK。
・よく見ると制約が小さいので、クエリ毎に左右を愚直に見に行っても大丈夫。
"""

import sys

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N, Q = MAP()
    S = input()

    for _ in range(Q):
        l, r = MAP()
        l -= 1
        if S[:l].count(S[l]) or S[r:].count(S[r-1]):
            YES()
        else:
            NO()
