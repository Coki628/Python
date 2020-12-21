# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/tkppc4-2/editorial-B.pdf
・400点自力ならず。
・木の走査
・必要な情報を戻り値に持たせる。
・2,3分岐したらダメだよなーまでは考えたけど、
　1本道に2つ以上写真があった時にうまく数えないで返すとか、
　最初だけ2つ分岐OKにすればうまくいくのとか、
　色々条件分けうまくやるのが出来なかった。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
nodes =[[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)
C = LIST()

# 写真がある頂点かどうか
has_pic = [0] * N
for c in C:
    has_pic[c-1] = 1
# 探索開始位置は写真あり頂点ならどこでも
start = C[0] - 1

def rec(cur, prev):
    # 戻り値に、その先で写真のあったパスの数を持たせる
    res = 0
    for nxt in nodes[cur]:
        if nxt != prev:
            res += rec(nxt, cur)
    # 最初の頂点で、その先に写真ありパスが3つ以上あればNG
    if prev == -1:
        if res >= 3:
            return INF
        else:
            return res
    # 最初以外は2つ以上でNG
    elif res >= 2:
        return INF
    # 1つは問題ないのでそのまま
    elif res == 1:
        return res
    # この先に1つもなければ、今の頂点の有無を確認
    elif res == 0:
        return has_pic[cur]

if rec(start, -1) != INF:
    Yes()
else:
    print('trumpet')
