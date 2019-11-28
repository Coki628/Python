# -*- coding: utf-8 -*-

"""
参考：https://ferin-tech.hatenablog.com/entry/2017/12/29/230333
　　　https://atcoder.jp/contests/arc003/submissions/7466576
・グリッドBFS、二分探索、最小値の最大化
・近い方針まで出てたのに、自分でうまくまとめられなかったから残念。
・最後にWAハマってたのは、にぶたんの下限。0でいいと思ったけど、0.99冪乗がそのうちほぼ0になるから
　0は到達ケースとして考慮しないといけない。で、下限は-1にした。
・計算量は500*500=25万のグリッドにダイクストラのlogとにぶたんlogが付いて、、、
・全部コスト1の辺ならBFSでいいじゃん、てことでダイクストラをBFSに置き換えた版。
・これでもpythonTLE,pypyTLE&MLE。log1個取れてちょっと速くはなったけどね。。
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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(40):
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

H, W = MAP()
grid = build_grid(H, W, '#', str, space=False, padding=True)
sh = sw = gh = gw = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == 's':
            sh = i; sw = j
        elif grid[i][j] == 'g':
            gh = i; gw = j

# 0.99の冪乗は事前計算しておく
pow099 = [0] * (H*W+1)
pow099[0] = 1
for i in range(1, H*W+1):
    pow099[i] = pow099[i-1] * 0.99

# グリッドBFS(コストは普通に時間を持って、明るさがmに満たない場所は壁みたいに行けなくする)
def bfs(m):
    from collections import deque

    numbers = {'1','2','3','4','5','6','7','8','9'}
    directions = ((1,0),(-1,0),(0,1),(0,-1))
    memo = list2d(H+2, W+2, INF)
    memo[sh][sw] = 0
    que = deque([(0, sh, sw)])

    while que:
        t, ch, cw = que.popleft()
        t += 1
        for direction in directions:
            dh, dw = direction
            nh = ch + dh
            nw = cw + dw
            # 数字マスだったら明るさ計算
            if grid[nh][nw] in numbers:
                light = int(grid[nh][nw]) * pow099[t]
                # 明るさが足りていれば経路候補になる(足りなければ壁扱い)
                if light >= m:
                    if memo[nh][nw] == INF:
                        memo[nh][nw] = t
                        que.append((t, nh, nw))
            # ゴールマスはキューに詰めないで記録のみ
            elif grid[nh][nw] == 'g':
                memo[nh][nw] = t
    # ゴール地点への到達時間を返却
    return memo[gh][gw]

# 明るさで二分探索して最大値を探す
res = bisearch_max(-1.0, 9.0, lambda m: bfs(m) != INF)
if res == -1.0:
    print(-1)
else:
    print(res)
