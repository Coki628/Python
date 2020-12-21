# -*- coding: utf-8 -*-

"""
・自力AC！ちょっと時間かかったけど。
・二分探索応用
・数学っぽい雰囲気の問題だったけど、最終的に式変形とかはほぼなかった。
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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

T1, T2 = MAP()
A1, A2 = MAP()
B1, B2 = MAP()

# a,bの状態からT1+T2の操作を行った後のa,bと、この操作での交差回数を返す
def calc(a, b):
    cnt = 0
    a1 = a + T1 * A1
    b1 = b + T1 * B1
    if a > b and a1 <= b1 or a < b and a1 >= b1:
        cnt += 1
    a2 = a1 + T2 * A2
    b2 = b1 + T2 * B2
    if a1 > b1 and a2 <= b2 or a1 < b1 and a2 >= b2:
        cnt += 1
    return a2, b2, cnt

# x+1回目の操作でまだ交差するか
def check(x):
    a = (T1 * A1 + T2 * A2) * x
    b = (T1 * B1 + T2 * B2) * x
    _, _, cnt = calc(a, b)
    return cnt >= 1

# 初回で同値に飛んだらその先もずっと繰り返す
a, b, cnt = calc(0, 0)
if a == b:
    print('infinity')
    exit()

# 最後に交差する直前までの操作回数を調べる
res = bisearch_max(-1, 10**18, check)
# 初回でも交差しなければ0
if res == -1:
    print(0)
    exit()

# 直前までの操作では、初回は1回、以降は2回ずつ交差する
ans = max(0, res * 2 - 1)
# 最後に交差する直前までa,bを移動させる
a = (T1 * A1 + T2 * A2) * res
b = (T1 * B1 + T2 * B2) * res
# 最後が1回か2回か確認
_, _, cnt = calc(a, b)
ans += cnt
print(ans)
