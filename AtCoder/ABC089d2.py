# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・ダブリング
・絶対想定解これじゃないなと思いつつも良い方針が出ずこれで通した。
・当時のがすんなり解いたっぽいのは微妙だけど、まあ色々なことが出来るようになってるのは良いことか。。
・計算量はH*W*log=9万logだけど、やっぱダブリングのlogは重いっぽい。
　pythonTLEでpypyAC0.6秒。
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

H, W, D = MAP()
grid = build_grid(H, W, 0, int)
HW = {}
for i in range(H):
    for j in range(W):
        HW[grid[i][j]-1] = (i, j)

# nxt[k][i] := iからスタートして、2^k回移動した後にいる位置
nxt = list2d(17, H*W, -1)
# cost[k][i] := iからスタートして、2^k回移動した時にかかるコスト
cost = list2d(17, H*W, INF)
for h in range(H):
    for w in range(W):
        cur = grid[h][w] - 1
        if cur + D < H * W:
            h2, w2 = HW[cur+D]
            # 1つ先の移動先とコストを記録
            nxt[0][cur] = cur + D
            cost[0][cur] = abs(h-h2) + abs(w-w2)

# ダブリングのテーブル構築
for k in range(1, 17):
    for i in range(H*W):
        if nxt[k-1][i] == -1:
            continue
        nxt[k][i] = nxt[k-1][nxt[k-1][i]]
        cost[k][i] = cost[k-1][i] + cost[k-1][nxt[k-1][i]]

for i in range(INT()):
    l, r = MAP()
    l -= 1; r -= 1
    cnt = (r - l) // D
    cur = l
    ans = 0
    # cnt回先に進む時のコストを確認
    for k in range(16, -1, -1):
        if cnt & 1<<k:
            ans += cost[k][cur]
            cur = nxt[k][cur]
    print(ans)
