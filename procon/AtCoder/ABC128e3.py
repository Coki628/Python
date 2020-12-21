# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc128/editorial.pdf
　　　https://atcoder.jp/contests/abc128/submissions/5657968
・イベントソート、各状態変化をイベントとして順番に処理していく。
・setとheapqをうまく組み合わせると、動的追加と削除+最小値取得が可能。
・(N+Q)logN=40万log20万=40万*約18=720万、python1.8秒ACでpypyはなぜかTLE。
"""

import sys
from heapq import heappush, heappop

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

N, Q = MAP()
# 3種類の状態変化をイベントとして一括管理する
# 0: 工事終了, 1: 工事開始, 2: 人の出発
events = []
for i in range(N):
    s, t, x = MAP()
    # (開始座標に合わせた状態の工事開始時間, フラグ, 実際の座標)
    events.append((s-x, 1, x))
    # (開始座標に合わせた状態の工事終了時間, フラグ, 実際の座標)
    events.append((t-x, 0, x))
for i in range(Q):
    d = INT()
    # (人の出発時間, フラグ, index)
    events.append((d, 2, i))
# 時系列に沿ってソート、フラグも同時刻な時先に処理して欲しいものから番号を振ってある
events.sort()

# setとheapqを合わせて、時刻に合わせた処理と最小値取得を行う
S = set()
hp = []
ans = [0] * Q
for t, flag, x in events:
    # 工事終了
    if flag == 0:
        S.remove(x)
    # 工事開始
    elif flag == 1:
        S.add(x)
        heappush(hp, x)
    # 人の移動
    elif flag == 2:
        # heapqの先頭要素ついて、setと状態を合わせる
        while len(hp) and hp[0] not in S:
            heappop(hp)
        if len(hp):
            ans[x] = hp[0]
        else:
            ans[x] = -1
[print(a) for a in ans]
