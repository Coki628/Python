# -*- coding: utf-8 -*-

"""
参考：https://tokumini.hatenablog.com/entry/2018/07/30/101439
　　　https://img.atcoder.jp/agc017/editorial.pdf
・公式解の方針
・足す回数、引く回数を固定すれば、その回数で自由に動ける範囲は定まる。
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

N, A, B, C, D = MAP()

# Aを起点に対称に広がるのでとりあえずAを起点とする
B = abs(A - B)
for i in range(N):
    # i回足して、j回引く
    j = N - i - 1
    # 小さく足して大きく引く
    mn = C * i - D * j
    # 大きく足して小さく引く
    mx = D * i - C * j
    # 範囲内なら自由に調節が効くのでOK
    if mn <= B <= mx:
        YES()
        exit()
NO()
