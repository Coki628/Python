# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2019/12/29/051900
・自力ならず。。これはまあしょうがないか。
・イベントソート
・色々条件整えてから、座標順にコストの増減を見ていく。
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

N, W, C = MAP()
events = []
for i in range(N):
    l, r, p = MAP()
    # (座標位置(半開区間にしておく), l or r, コスト)
    events.append((l+1, 0, p))
    # 終端からC離れたら、その石はもう動かさなくていい
    events.append((r+C, 1, p))

# 番兵2つ
events.append((C, 0, 0))
events.append((INF, 0, 0))
events.sort()

ans = INF
cost = 0
l = 0; r = 1
prev = -1
# 今見ている位置を区間Cの右端とする
for x, lr, p in events:
    # 座標位置が変わったら、直前の座標についてコスト確認
    if prev >= C and prev != x:
        ans = min(ans, cost)
    if x > W:
        break
    if lr == l:
        cost += p
    else:
        cost -= p
    prev = x
print(ans)
