# -*- coding: utf-8 -*-

"""
・自力AC
・問題の理解に時間食ったけど、分かれば実装やるだけ。
　リストの連結をうまくやると、間に入る記号もうまく収まる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
N = 2**N

ans = []
for i in range(N):
    *Q, R = LIST()
    if R:
        tmp = []
        for i, q in enumerate(Q):
            if q:
                tmp.append('P_{0}'.format(i+1))
            else:
                tmp.append('¬P_{0}'.format(i+1))
        ans.append('({0})'.format('∧'.join(tmp)))
if len(ans) == N:
    print('A=⊤')
elif len(ans) == 0:
    print('A=⊥')
else:
    print('A={0}'.format('∨'.join(ans)))
