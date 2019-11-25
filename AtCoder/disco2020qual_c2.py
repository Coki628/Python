# -*- coding: utf-8 -*-

"""
・400点自力ならず。。
・何もない行は上か下から丸コピでいいってのが出なかった。考えてみればそうだよね。。
・最初は各行で処理しようとしてたから、その方針のまま行けばよかった。。
・ツイートで上記ヒントを得てAC。
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

H, W, K = MAP()

grid = build_grid(H, W, '*', str, space=False, padding=True)

ans = list2d(H+2, W+2, -1)
cnt = 0
for i in range(1, H+1):
    # 苺が1つもない行は1つ前をそのまま持ってくる
    if ''.join(grid[i]).count('#') == 0:
        ans[i] = ans[i-1]
        continue
    for j in range(1, W+1):
        if grid[i][j] == '#':
            cnt += 1
        ans[i][j] = cnt
    # その行で最初の苺が出現する前の場所を塗り直す
    for j in range(W, 0, -1):
        if ans[i][j] == 0 or ans[i][j] == ans[i-1][W]:
            ans[i][j] = ans[i][j+1]

# 先頭行から何もなかった場合は後の行からそのまま持ってくる
for i in range(H, 0, -1):
    if ans[i][1] == -1:
        ans[i] = ans[i+1]

for i in range(1, H+1):
    print(*ans[i][1:W+1])
