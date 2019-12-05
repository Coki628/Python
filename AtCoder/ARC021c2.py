# -*- coding: utf-8 -*-

"""
・自力AC！
・二分探索、式変形、等差数列の和
・式変形の詳細は自分ノート参照。
・しっかり想定解の方針に辿り着けたので、良かったと思う。
・1回WAで、？ってなったけど、K回超えてる可能性があるのにすぐ気づいて調整できた。
・計算量は10万のNにlogついて、pypyAC0.4秒。
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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def get_sum(a, b, c):
    """ 等差数列の和：(初項a, 末項b, 項数c) """
    return (a+b) * c // 2

K = INT()
N = INT()
AD = []
for i in range(N):
    a, d = MAP()
    AD.append((a, d))

# 各家を価格m以内まで増築する時、増築回数がK回以上になるか
def calc(m):
    cnt = 0
    for a, d in AD:
        if m - a >= 0:
            cnt += ((m-a) // d) + 1
    return cnt >= K

# K回以上増築可能な最小価格resを取得
res = bisearch_min(0, 10**18, calc)
ans = 0
sm = 0
for a, d in AD:
    if res - a >= 0:
        # 最小価格resまでの増築回数
        cnt = ((res-a) // d) + 1
        # それにかかる費用は等差数列の和で出せる
        ans += get_sum(a, a+d*(cnt-1), cnt)
        sm += cnt
# Kを超えた回数分の調整
while sm > K:
    ans -= res
    sm -= 1
print(ans)
