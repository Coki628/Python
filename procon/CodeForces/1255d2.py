# -*- coding: utf-8 -*-

"""
・コンテスト終了後ほぼ自力AC
・BFSとかまじいらなかった。。各行で左右の進行方向交互にするだけで、前から普通に埋めれた。
・これは参戦中に通したかったなー。。
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
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
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

for _ in range(INT()):

    H, W, K = MAP()
    grid = build_grid(H, W, '*', str, space=False, padding=False)

    # 各鶏にいくつずつRを割り振るかを出しておく
    rcnt = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'R':
                rcnt += 1
    p = rcnt // K
    q = rcnt % K

    # 各鶏kで答えのグリッドを埋めていく
    ans = list2d(H, W, 0)
    cnt = k = 0
    for i in range(H):
        for j in range(W):
            # q匹までは余り分の+1を含む
            x = p + 1 if k < q else p
            # 領域が連続するように、奇数行目は右→左に進む
            if i % 2 == 1:
                j = W - j - 1
            ans[i][j] = k
            if grid[i][j] == 'R':
                cnt += 1
            # Rの数が予定量に達したら次の鶏(最後の1匹は最後までやりきる)
            if cnt == x and k != K - 1:
                cnt = 0
                k += 1

    # 出力用に数値を必要な文字に変換
    for i in range(H):
        for j in range(W):
            # 数字(そのまま)
            if ans[i][j] < 10:
                ans[i][j] = str(ans[i][j])
            # 英小文字
            elif 10 <= ans[i][j] < 36:
                ans[i][j] = chr(ans[i][j]+87)
            # 英大文字
            else:
                ans[i][j] = chr(ans[i][j]+29)
    for i in range(H):
        print(''.join(ans[i]))
