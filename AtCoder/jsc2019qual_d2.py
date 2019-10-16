# -*- coding: utf-8 -*-

"""
参考：https://maspypy.com/atcoder-参加感想-2019-08-25jsc2019予選
　　　https://img.atcoder.jp/jsc2019-qual/editorial.pdf
・奇数の閉路がNG、までは分かったけど、確かにそれは二部グラフだね。。
・完全グラフを二部グラフに分割していく。
・公式解とかビットで処理してたけど、twitterで誰かが再帰とか分割統治法言ってたのを参考に実装。
・前の回で同じグループにいた頂点同士は辺張れてないはずだから、
　次の回でそいつらをまた半分にして二部グラフ構築、っていうのは理に適ってるな。
"""

import sys

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

N = INT()

G = list2d(N, N, 0)
A = list(range(N))

def rec(li, lv):
    n = len(li)
    # 頂点が1つ以下になったら終了
    if n <= 1:
        return
    # 頂点集合を半分に分けて、2部グラブにする
    li1 = li[:n//2]
    li2 = li[n//2:]
    for u in li1:
        for v in li2:
            G[u][v] = lv
    # 半分にした集合を再帰的に処理する
    rec(li1, lv+1)
    rec(li2, lv+1)
    return

rec(A, 1)

ans = []
for i in range(N):
    row =[]
    for j in range(i+1, N):
        row.append(G[i][j])
    ans.append(row)

for row in ans:
    print(*row)
