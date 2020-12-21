# -*- coding: utf-8 -*-

"""
・400点自力AC！
・二分探索
・連続した長さの数え方ミスってて時間かかった。。
・いくつを達成できるか、で二分探索する視点にちゃんと気付けたのは良かった。
・二分探索をライブラリ化した版
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(li): return reduce(lcm, li, 1)
def gcd_list(li): return reduce(gcd, li, 0)
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

N, K = MAP()
A = LIST()

# 連続した長さ毎にまとめる
cnt = 1
B = []
for i in range(1, N):
    if A[i-1] == A[i]:
        cnt += 1
    else:
        B.append(cnt)
        cnt = 1
B.append(cnt)

# K回以内の操作で退屈さmを実現できるか
def calc(m):
    cnt = 0
    for b in B:
        if b > m:
            cnt += b//(m+1)
    return cnt <= K

# 二分探索
print(bisearch_min(0, max(B), calc))
