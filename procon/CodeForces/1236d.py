# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70654
・グリッド、二分探索
・行列毎に障害物の位置を把握できるように、隣接リストっぽく持たせる。
・この問題、10万log(+多分定数倍もそこそこ)で制約が1秒だからか、pypyでもTLE。
・これと同じだとC++でもTLEで、多分訪問済チェックのwhileがたくさん回ってるから。
・そっちも二分探索できるようにして、色々細かい試行錯誤があって、C++はAC。
"""

import sys
from bisect import bisect_left

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

H, W, K = MAP()
rows = [[] for i in range(H)]
cols = [[] for i in range(W)]
for i in range(K):
    r, c = MAP()
    r -= 1
    c -= 1
    # 障害物の座標を、行列毎に隣接リストっぽく持たせる
    rows[r].append(c)
    cols[c].append(r)
for i in range(H):
    # 前後に番兵を入れると、この後二分探索で管理がしやすい
    rows[i].append(-1)
    rows[i].append(W)
    rows[i].sort()
for i in range(W):
    cols[i].append(-1)
    cols[i].append(H)
    cols[i].sort()

cur = [0, 0]
cnt = 1
visitedr = set()
visitedc = set()

def right():
    r, c = cur[0], cur[1]
    visitedr.add(r)
    # 次の障害物の位置
    idx = bisect_left(rows[r], c)
    # その手前まで進める
    nxt = rows[r][idx] - 1
    # 既に訪れた列ならずらす
    while nxt in visitedc:
        nxt -= 1
    # 動けなかったら終了
    if nxt <= c:
        return False
    else:
        cur[1] = nxt
        global cnt
        cnt += nxt - c
        return True


def down():
    r, c = cur[0], cur[1]
    visitedc.add(c)
    idx = bisect_left(cols[c], r)
    nxt = cols[c][idx] - 1
    while nxt in visitedr:
        nxt -= 1
    if nxt <= r:
        return False
    else:
        cur[0] = nxt
        global cnt
        cnt += nxt - r
        return True

def left():
    r, c = cur[0], cur[1]
    visitedr.add(r)
    idx = bisect_left(rows[r], c) - 1
    nxt = rows[r][idx] + 1
    while nxt in visitedc:
        nxt += 1
    if nxt >= c:
        return False
    else:
        cur[1] = nxt
        global cnt
        cnt += c - nxt
        return True

def up():
    r, c = cur[0], cur[1]
    visitedc.add(c)
    idx = bisect_left(cols[c], r) - 1
    nxt = cols[c][idx] + 1
    while nxt in visitedr:
        nxt += 1
    if nxt >= r:
        return False
    else:
        cur[0] = nxt
        global cnt
        cnt += r - nxt
        return True

while True:
    if not right(): break
    if not down(): break
    if not left(): break
    if not up(): break

# 踏破したマス数 == 全体 - 障害物 ならOK
if cnt == H*W-K:
    Yes()
else:
    No()
