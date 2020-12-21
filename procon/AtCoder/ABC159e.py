# -*- coding: utf-8 -*-

"""
・終了約10分後、自力AC。悔しい。。
・ビット全探索、累積和
・ほぼ方針は合ってたのに、K個以内にできないケースでwhileがループ抜けられないのに
　気づいたのが終了後。そこ直したら通った。pypyAC0.4秒。
・速度が足りないのかと思って、にぶたんのlog削ったりした時間で気づいていれば。。
"""

import sys
from itertools import accumulate

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

def popcount(i):
    return bin(i).count('1')

H, W, K = MAP()
grid = build_grid(H, W, 0, int, space=0)

ans = INF
# 行の切り方は2^(H-1)通り全部試す
for S in range(1<<(H-1)):
    # H行をN行にまとめる(N-1箇所切る)
    N = popcount(S) + 1
    A = list2d(N, W, 0)
    for w in range(W):
        A[0][w] = grid[0][w]
    h = 0
    for k in range(H-1):
        if S & 1<<k:
            h += 1
        for w in range(W):
            A[h][w] += grid[k+1][w]
    # 累積和
    for i in range(N):
        A[i] = [0] + list(accumulate(A[i]))
    cnt = 0
    cur = 1
    while 1:
        # 各行なるべく先まで伸ばして、一番短かった行に合わせて進む
        nxt = INF
        for i in range(N):
            j = cur
            tgt = K + A[i][cur-1]
            while j < W+1 and A[i][j] <= tgt:
                j += 1
            nxt = min(nxt, j)
        # 1列も進めない時があったらこの切り方はNG
        if cur == nxt:
            cnt = INF
            break
        cur = nxt
        # 最後まで来たらループを抜ける
        if cur == W+1:
            break
        cnt += 1
    # 行を切った回数(N-1) + 列を切った回数(cnt)
    ans = min(ans, N-1+cnt)
print(ans)
