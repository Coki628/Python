"""
・自力AC
・グリッド、実装、15パズル、再帰探索、バックトラック
・本当の15パズルは確か普通に再帰全探索やると死ぬんだったはずだけど、
　これは1つのピースにつき1度しか動かせない制約があるので、なら相当手数減るだろう、と
　再帰全探索で愚直シミュレーション。実装は頑張る。
・スワップの状態とか使用済チェックは再帰が戻ってきた時に状態を戻す。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

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

target = build_grid(4, 4, -1, int, padding=1)
grid = list2d(6, 6, -1)
num = 1
for i in range(1, 5):
    for j in range(1, 5):
        grid[i][j] = num
        num += 1
grid[4][4] = 0
src = (4, 4)

# 目標配置と一致しているかのチェック
def check():
    for i in range(1, 5):
        for j in range(1, 5):
            if grid[i][j] != target[i][j]:
                return False
    return True

used = [0] * 16
directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
def rec(h, w):
    if check():
        return True
    for dh, dw in directions:
        h2 = h + dh
        w2 = w + dw
        if grid[h2][w2] == -1:
            continue
        if used[grid[h2][w2]]:
            continue
        # 使用済の更新
        used[grid[h2][w2]] = 1
        # スワップ
        grid[h][w], grid[h2][w2] = grid[h2][w2], grid[h][w]
        if rec(h2, w2):
            return True
        # 戻す時はスワップが先
        grid[h][w], grid[h2][w2] = grid[h2][w2], grid[h][w]
        # 未使用に戻す
        used[grid[h2][w2]] = 0
    return False

if rec(*src):
    Yes()
else:
    No()
