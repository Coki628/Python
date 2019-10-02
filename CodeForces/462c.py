# -*- coding: utf-8 -*-

"""
・蟻本演習
・貪欲、二分木、ハフマン符号、優先度付きキュー
・結果として与えられる数列に対して、2つずつ足し合わせていって最後の1つになるまでの過程は、二分木と考えることができる。
・この二分木を最適に遡って根まで到達するには、優先度付きキューを使うと効率良くできる。
・蟻本の例及びハフマン符号はこれの最小化、この問題では最大化する。
"""

import sys
from heapq import heappush, heappop, heapify

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

N = INT()
A = LIST()

A = [-a for a in A]
heapify(A)

ans = 0
# 要素が1つになるまで(二分木の根に辿り着くまで)足し合わせていく
while len(A) > 1:
    # 大きい方から処理していく
    a = heappop(A)
    b = heappop(A)
    ans += -(a+b)
    heappush(A, a+b)
ans += -A[0]
print(ans)
