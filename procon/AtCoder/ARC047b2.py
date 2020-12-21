# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc047-57126901
・座標変換、それに気付けたのは良かったけど、うまく扱えなかった。
・実際は各点が存在している場所は、答えの点を中心としたどこかの正方形に固定できるので、それを探す。
・なんかすごい変な位置に点があったらすごい頑張らないと距離同じにできないんじゃないかとか
　色々考えてしまったけど、そういうのはないと考えて良かったみたいだね。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N = INT()
XY = []
U, V = [], []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))
    # 座標変換：斜めに向かうuv座標を作っておく
    U.append(x+y)
    V.append(x-y)

mxu, mnu = max(U), min(U)
mxv, mnv = max(V), min(V)
ulen = mxu - mnu
vlen = mxv - mnv
D = max(ulen, vlen)
ans = set()
# 縦横の長い方は真ん中で決まり、短い方はどっちかに寄せる
u1 = mxu - D // 2
u2 = mnu + D // 2
v1 = mxv - D // 2
v2 = mnv + D // 2
# 計4パターン(実際は長い方は重複してるから1~2パターン)
ans.add((u1, v1))
ans.add((u1, v2))
ans.add((u2, v1))
ans.add((u2, v2))

for u, v in ans:
    if (u+v)%2 == 0: 
        # 変換したuv座標をxyに戻す
        x = (u+v) // 2
        y = u - x
        dist = set()
        for xi, yi in XY:
            dist.add(abs(x-xi)+abs(y-yi))
        # 全頂点での距離が一致していればOK
        if len(dist) == 1:
            print(x, y)
            exit()
