# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc137/editorial.pdf
・公式解、優先度付きキュー
・やってみればこれはすごくシンプル。自力出なかったの残念。。
・計算量は10万の多分NlogN系だけど、やっぱセグ木より全然速くてpythonで0.4秒AC。
"""

import sys
from heapq import heappush, heappop

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

N, M = MAP()
# 日数別に報酬の値を格納
days = [[] for i in range(10**5+1)]
for i in range(N):
    a, b = MAP()
    days[a].append(b)

que = []
ans = 0
# 制約の厳しい後半から確認していく
for i in range(M-1, -1, -1):
    a = M-i
    # 残日数aで使えるようになるタスクを追加
    for b in days[a]:
        heappush(que, (-b, a))
    # キューに中身があったら最大のものを取得
    if len(que):
        b, a = heappop(que)
        ans += -b
print(ans)
