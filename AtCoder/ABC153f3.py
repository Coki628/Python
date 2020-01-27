# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/mugen_1337/status/1221686960856547328
・公式に近い方針もやっておく。
・いもす法、ただし先に全部の点打ち込んで累積和じゃなくて、
　前に進みながら1つずつ累積和と点の打ち込みをする。
・後の全体の流れはセグ木で殴るのとそんなに違いはない。
・さすがに区間加算やるより速くて、pypyAC0.8秒。
"""

import sys
from bisect import bisect_right
from operator import itemgetter

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

N, D, K = MAP()

XH = []
for i in range(N):
    x, h = MAP()
    XH.append((x, h))

XH.sort(key=itemgetter(0))
X, _ = zip(*XH)
D = D * 2

imos = [0] * (N+1)
ans = 0
for i, (x, h) in enumerate(XH):
    # いもすの累積和
    imos[i+1] += imos[i]
    # 前回までの爆撃でのダメージを引く
    h -= imos[i] * K
    if h <= 0:
        continue
    # 今回必要な爆撃回数
    cnt = ceil(h, K)
    # 今回の爆撃が届く右端
    idx = bisect_right(X, x+D)
    # 今回分のいもすの始点と終点を打ち込む
    imos[i+1] += cnt
    imos[idx] -= cnt
    ans += cnt
print(ans)
