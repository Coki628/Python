# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・どこまでを前半にしてどこまでを後半にするかをN~N*2まで決め打ちすれば、
　前半から小さい0個と後半から大きいN個除く、前半から小さい1個と後半から大きいN-1個除く、
　みたいに、どの範囲から小さいのと大きいのをいくつ除くかが定まるから、うまく進められる。
・1度に変化する要素は前半後半高々1個ずつなので、計算量的にも大丈夫。
・特に後半部分の操作がゴチャゴチャするので、heapqとsetでうまくデータを持つ。
・前のやつ見たけど、確かに前半後半完全に独立にやって、最後に合流させて比較、
　のがスッキリまとまるな。それだと後半部分も複雑なことしなくて済む。
・まあこれでもちゃんとAC取れてるしいいんだけど。
"""

import sys
from heapq import heapify, heappush, heappop

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
A = LIST()

A1 = A[:N]
A2 = A[N:]
sm1 = sum(A1)
sm2 = sum(A2)
que1 = []
que2 = []
S = set()
# 前半部分準備
for i, a in enumerate(A1):
    que1.append((a, i))
heapify(que1)
# 後半部分準備
A2 = [(a, i) for i, a in enumerate(A2, N)]
A2.sort(reverse=1)
for a, i in A2[:N]:
    sm2 -= a
    # 除いた要素をsetとheapqで持っておく
    S.add(i)
    que2.append((a, i))
heapify(que2)

ans = sm1 - sm2
# どこまでを前半・後半とするかでN~N*2までを試す
for i in range(N, N*2):
    # 前半にiを追加
    heappush(que1, (A[i], i))
    sm1 += A[i]
    # 前半から1つ除く
    a, _ = heappop(que1)
    sm1 -= a
    # iが取り除かれた要素なら後半は何もしないでOK
    if i not in S:
        # 後半からiを削除
        sm2 -= A[i]
        while 1:
            # 後半から除いていた要素を1つ戻す
            a, j = heappop(que2)
            S.remove(j)
            # 位置的にまだ残っている要素が来たらそれを使う
            if j > i:
                sm2 += a
                break
    ans = max(ans, sm1 - sm2)
print(ans)
