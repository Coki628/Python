# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC！にするつもりがWA踏んでもたついた。
・結局それはただのケアレスミス。ちゃんと気づけたからいいけど。
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

def check(l, r):
    # 左だけ行く
    if r <= 0:
        return -l
    # 右だけ行く
    elif l >= 0:
        return r
    # 両方行く
    else:
        # 短い方を先に行って往復
        return min(-l, r) * 2 + max(-l, r)

N, K = MAP()
A = LIST()

ans = INF
for i in range(N):
    if i+K > N:
        break
    l = A[i]
    r = A[i+K-1]
    ans = min(ans, check(l, r))
print(ans)
