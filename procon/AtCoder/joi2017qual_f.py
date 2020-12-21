# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2017/2018-yo/2018-yo-t6-review.html
　　　https://www.ioi-jp.org/joi/2017/2018-yo/2018-yo-t6-sample.cpp
・蟻本演習3-2-2
・尺取法、条件を満たす区間の組の数え上げ
・尺取パート自体は、基本的なものに近い。
・特徴としては、条件を満たした時に自分より右をまとめて足すって感じ。
・二分探索
・小さい方からL番目がX => X以下の数がL以上となる最小値を探す。
・上記言い換えは典型のようで、全く同じ考え方はARC037cにも出てきた。
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

N, K, L = MAP()
A = LIST()
MAX = 2 * 10**5

# x以下の数(がK番目となる区間の数)がL個以上かどうか
def check(x):
    # 尺取法
    l = r = ans = 0
    # cnt：x以下の数の個数
    cnt = 0
    while l < N:
        while r < N and cnt < K:
            if A[r] <= x:
                cnt += 1
            r += 1
        # K個になった時は、現在の左端とのペアで自分より右の右端は全て条件を満たすので、まとめて足す
        if cnt == K:
            ans += N - r + 1
        if l == r:
            # 左が右に追いついたら、右も左に合わせて+1する
            r += 1
        else:
            # 左を動かす分、合計から引く
            if A[l] <= x:
                cnt -= 1
        l += 1
    return ans >= L

print(bisearch_min(0, MAX, check))
