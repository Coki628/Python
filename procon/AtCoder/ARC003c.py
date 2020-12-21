# -*- coding: utf-8 -*-

"""
・自力WA。
・ダイクストラ内で色々ゴニョゴニョやったけどダメだった。
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

H, W = MAP()
grid = build_grid(H, W, '#', str, space=False, padding=True)
sh = sw = gh = gw = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == 's':
            sh = i; sw = j
        elif grid[i][j] == 'g':
            gh = i; gw = j

# グリッドダイクストラ(コストは距離じゃなく明るさで、最大を取っていく)
def dijkstra():
    from heapq import heappush, heappop

    numbers = {'1','2','3','4','5','6','7','8','9','0'}
    directions = ((1,0),(-1,0),(0,1),(0,-1))
    memo = list2d(H+2, W+2, -1)
    memo[sh][sw] = INF
    que = [(-INF, sh, sw, 0)]

    while que:
        cost, ch, cw, t = heappop(que)
        cost = -cost
        t += 1
        for direction in directions:
            dh, dw = direction
            nh = ch + dh
            nw = cw + dw
            # 数字マスだったらコスト計算
            if grid[nh][nw] in numbers:
                # これまで通った中で最小の明るさ
                nxt = min(cost, int(grid[nh][nw]) * 0.99**t)
                # メモより良い状態なら記録してキューに詰める
                if nxt > memo[nh][nw]:
                    memo[nh][nw] = nxt
                    heappush(que, (-nxt, nh, nw, t))
            # ゴールマスはキューに詰めないで記録のみ
            elif grid[nh][nw] == 'g':
                memo[nh][nw] = max(memo[nh][nw], cost)
    # ゴール地点への最大の明るさを返却
    return memo[gh][gw]

res = dijkstra()
print('{:.10f}'.format(res))
