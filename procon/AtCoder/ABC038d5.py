# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/e77685614f3c6bf86f44
　　　https://ameblo.jp/linuxojisan009/entry-12433426997.html
・別解
・公式はBITでmax取ってく感じだったけど、LISでさくっと解けることが判明。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def lenLIS(A: list, equal=False):
    """ 最長増加部分列の長さ """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    # デフォルトは狭義のLIS(同値を含まない)
    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            # Lの末尾よりaが大きければ増加部分列を延長できる
            L.append(a)
        else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect(L, a)] = a
    return len(L)

N = INT()
HW = []
for i in range(N):
    w, h =MAP()
    # 紛らわしいので縦横の順にしておく
    HW.append((h, w))

# 縦の昇順、横の降順に並べておく
HW.sort(key=itemgetter(1), reverse=True)
HW.sort(key=itemgetter(0))

# 横でLIS
H, W = zip(*HW)
print(lenLIS(W))
