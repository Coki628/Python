# -*- coding: utf-8 -*-

"""
・自力AC、、ではないかな。(サンプルのみACで泣いてテストケースだけ見てしまった)
・六角形のグリッド(indeednow2015finala_bと扱いはほぼ同じ)
・割と方針はすぐまとまったのに、ACするまでに色々詰まって時間かかってしまった。
・最初、各マスから外に出られるか見に行こうとしてたんだけどこれが良くなかった。
・実際は外周マスから内側に向かって到達可能かBFSなりすれば良かった。
　(自力で気付けたのは良かったけど、相当時間経って色々やり尽くしてから。)
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

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

# 斜めの方向を追加
directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1,-1), (-1, -1), (1, 1), (-1, 1))
def dfs(h, w):
    if isout[h][w] == 1:
        return isout[h][w]
    isout[h][w] = 1
    for i, (dh, dw) in enumerate(directions):
        # 奇数行は左上、左下は行かない
        if h % 2 == 1 and i in [4, 5]:
            continue
        # 偶数行は右上、右下は行かない
        if h % 2 == 0 and i in [6, 7]:
            continue
        h2 = h + dh
        w2 = w + dw
        # 建物がある or グリッド外
        if grid[h2][w2] == 1 or grid[h2][w2] == -1:
            continue
        dfs(h2, w2)

W, H = MAP()
grid = build_grid(H, W, -1, int, padding=1)

# 外周全てのマスから、到達可能な場所を調べる
isout = list2d(H+2, W+2, -1)
for h in range(1, H+1):
    for w in range(1, W+1):
        isout[h][w] = 0
for h in range(1, H+1):
    if grid[h][1] == 0:
        dfs(h, 1)
    if grid[h][W] == 0:
        dfs(h, W)
for w in range(1, W+1):
    if grid[1][w] == 0:
        dfs(1, w)
    if grid[H][w] == 0:
        dfs(H, w)

ans = 0
for h in range(1, H+1):
    for w in range(1, W+1):
        # 建物があるマスについて、外に出られるマスに面しているかを確認
        if grid[h][w] == 1:
            for i, (dh, dw) in enumerate(directions):
                if h % 2 == 1 and i in [4, 5]:
                    continue
                if h % 2 == 0 and i in [6, 7]:
                    continue
                h2 = h + dh
                w2 = w + dw
                # 6方向見て、外に出られるマスに面していたらこの壁面は使う
                if isout[h2][w2] == 1 or isout[h2][w2] == -1:
                    ans += 1
print(ans)
