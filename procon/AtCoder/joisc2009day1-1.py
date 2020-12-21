# -*- coding: utf-8 -*-

"""
・自力AC
・(メモしない)再帰探索
・なんとなくルンルンがパティいっぱい食べるやつ思い出した。
・再帰で解くやつ、割と苦手意識があったからちゃんと通せて良かった。
　まあ、これは見た目的に再帰しなさいがあからさまって感じだけどね。。
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
MOD = 10 ** 5

N, K = MAP()

def rec(size, row):
    if size == 1:
        return 'J'
    nxtsize = size // 2
    # 欲しい行が上半分か下半分かで場合分け
    if row <= nxtsize:
        return 'J' * nxtsize + 'O' * nxtsize
    else:
        return 'I' * nxtsize + rec(nxtsize, row - nxtsize)

size = 2**N
ans = rec(size, K)
print(ans)
