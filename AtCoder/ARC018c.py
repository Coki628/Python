# -*- coding: utf-8 -*-

"""
・自力AC！
・グリッド系、縦横独立に処理
・縦移動は事前にいるべき行の集合を用意してから1マスずつ確認。
・横移動は移動してきた数と空席で足し引きして、そのマスに収まらない分を1マスずつ隣に動かしていく。
　(実際には人が動くとしても、空席を動かすとみなしても移動する数に影響ないので、人か席の余ってる方を動かしていく。)
・1回WA食らったけど、人を管理する時値で管理してたから、一意じゃない時があって集合に入れる時バグってたぽい。
　ソート前の元indexで管理するようにしたら無事AC。
・計算量は最大100万のDP配列をソートしてるから100万log=100万*約20で2000万、の割にはpypy余裕のAC0.2秒。
　前100万超の配列ソートしてTLE食らったことがあったけど、中身の内容も関係するのかな。
・方針、うまくまとめられたと思う。
・と思ったけど公式解はもっと賢い方法でやってるぽい。
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

H, W = MAP()
x, a, p = MAP()
N = H * W

# 現在の席順を構築
dp = [()] * N
dp[0] = (x, 0)
for i in range(1, N):
    dp[i] = ((dp[i-1][0] + a) % p, i)
grid = list2d(H, W, 0)
for i in range(H):
    for j in range(W):
        grid[i][j] = dp[i*W+j][0]

# 本来いるべき行の集合を作る
S = [set() for i in range(H)]
D = {}
row = 0
# 値は一意でない可能性があるので、元indexで集合を管理する
for i, (a, orgi) in enumerate(sorted(dp)):
    S[row].add(orgi)
    D[orgi] = row
    if i%W == W-1:
        row += 1

# 縦移動の処理
moved = list2d(H, W, 0)
vacant = list2d(H, W, 0)
ans = 0
for i in range(H):
    for j in range(W):
        orgi = i*W + j
        # いる行が違っていたら
        if orgi not in S[i]:
            nxt = D[orgi]
            # 自分の移動とここが空席になることを記録
            moved[nxt][j] += 1
            vacant[i][j] += 1
            # 縦移動の距離を答えに追加
            ans += abs(nxt-i)

# 横移動の処理
for i in range(H):
    cnt = 0
    for j in range(W):
        # 移動してきた人数と空席を比較して、このマスで余る数をカウント
        if moved[i][j] > vacant[i][j]:
            cnt += abs(moved[i][j] - vacant[i][j])
        elif moved[i][j] < vacant[i][j]:
            cnt -= abs(moved[i][j] - vacant[i][j])
        # 余る数を1つ横に動かすので、答えに追加
        ans += abs(cnt)
print(ans)
