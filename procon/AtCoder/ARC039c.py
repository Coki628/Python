# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc039
・自力ではさっぱり出なかったけど、新しい知識だからOK。
・グリッド、Dancing Links
・各マスについて、四方の未訪問情報を持たせる。
・このデータの持ち方はDancing Linksって言うらしいけど、初めてやった。
・訪問済じゃなくて、未訪問情報を持たせるっていうのも画期的だった。
・解説読んで、頑張って自力実装できたものの、これはMLE。。
"""

import sys
from collections import defaultdict

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
S = input()

# links[x][y] := 座標(x, y)から見た、四方の最初の未訪問位置
links = defaultdict(lambda: defaultdict(dict))
links[0][0] = {'L': -1, 'R': 1, 'D': -1, 'U': 1,}

# 四方の未訪問情報を持ったマスの作成
def create(x, y):
    # 今回のマスを作るための情報を四方から集める
    if links[x-1][y]:
        l = links[x-1][y]['L']
    else:
        l = x - 1
    if links[x+1][y]:
        r = links[x+1][y]['R']
    else:
        r = x + 1
    if links[x][y-1]:
        d = links[x][y-1]['D']
    else:
        d = y - 1
    if links[x][y+1]:
        u = links[x][y+1]['U']
    else:
        u = y + 1
    # 今回のマスが作成した情報を四方の未訪問手前のマスへ伝える
    links[l+1][y]['R'] = r
    links[r-1][y]['L'] = l
    links[x][d+1]['U'] = u
    links[x][u-1]['D'] = d
    return {'L': l, 'R': r, 'D': d, 'U': u,}

cur = (0, 0)
for s in S:
    x, y = cur
    if s == 'L' or s == 'R':
        # 次の未訪問位置
        nx = links[x][y][s]
        # 新しくマスを作成
        links[nx][y] = create(nx, y)
        cur = (nx, y)
    else:
        ny = links[x][y][s]
        links[x][ny] = create(x, ny)
        cur = (x, ny)
print(cur[0], cur[1])
