# -*- coding: utf-8 -*-

"""
・まあさくっと自力AC
・key,valueをどっちがどっちかちょっとこんがらがった。
"""

import sys
from operator import itemgetter

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
# {CD番号: 入っている場所}
D = {i: i for i in range(N+1)}
cur = 0
for i in range(M):
    nxt = INT()
    D[cur], D[nxt] = D[nxt], D[cur]
    cur = nxt

# 入っている場所でソートしてCD番号を出力(先頭以外)
for k, v in sorted(D.items(), key=itemgetter(1))[1:]:
    print(k)
