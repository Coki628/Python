# -*- coding: utf-8 -*-

"""
参考：http://tutuz.hateblo.jp/entry/2018/09/06/224654
・O(1)解法は思いつくか分からないから、別解もやる。
　これはO(logN)でAC解法。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, X = MAP()

# 最初のN以外は、平行四辺形を小さくしていくと考える
def rec(a, b):
    if b == 0:
        # 最後に割り切れた時は半分でいいので減らす
        return -a
    a, b = min(a, b), max(a, b)
    # 同じだけ小さくする所はまとめてやる
    return a*2*(b//a) + rec(a, b%a)

print(N+rec(X, N-X))
