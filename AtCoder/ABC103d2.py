# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc103/editorial.pdf
・速解き特訓ABC周回
・自力ならず。。1年半前にこれコンテスト中に通したのか。。。
・片方でソートしてもう片方をうまくやる系
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
edges = []
for i in range(M):
    a, b = MAP()
    edges.append((a, b))
# 右端でソート
edges.sort(key=itemgetter(1))

nxt = cnt = 0
for a, b in edges:
    # 左端が切る位置より先にあったらもう1回切る
    if a > nxt:
        # 次に切る位置
        nxt = b - 0.5
        cnt += 1
print(cnt)
