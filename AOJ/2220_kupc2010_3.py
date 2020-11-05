# -*- coding: utf-8 -*-

"""
参考：https://nanikaka.hatenadiary.org/entry/20121001/1349075090
・蟻本演習3-1-2
・小数の二分探索、微分、誤差
・色々試行錯誤の末、やっとAC。にぶたん100回だとダメで1000回にした。
・にぶたんの終了判定をwhileにした。EPS10**-9に対しての10**-12だとWAで、
　EPS10*-8にしたら通った。
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
EPS = 10 ** -8

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    while abs(ok-ng) > 10**-12:
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# 2次方程式の解の公式
def dim2formula(a, b, c):
    from math import sqrt

    try:
        return (-b+sqrt(pow(b, 2)-4*a*c))/(2*a), (-b-sqrt(pow(b, 2)-4*a*c))/(2*a)
    # 解なしはNoneを返却
    except ValueError:
        return (None, None)

# 3次→2次への微分
def calc1(a, b, c, d):
    a *= 3
    b *= 2
    return (a, b, c)

# ax^3+bx^2+cx+dを解く
def calc2(x):
    return a*x**3 + b*x**2 + c*x + d

# /のグラフ用
def check1(x):
    res = calc2(x)
    return res < 0

# \のグラフ用
def check2(x):
    res = calc2(x)
    return res > 0

for _ in range(INT()):
    a, b, c, d = MAP()
    ans = [0] * 2

    a2, b2, c2 = calc1(a, b, c, d)
    ans1, ans2 = dim2formula(a2, b2, c2)

    # 微分して解いた結果が解なしなら、元の式は極値なしで解は1つ
    if ans1 is None:
        if a > 0:
            # /のグラフ
            x = bisearch_max(-INF, INF, check1)
        else:
            # \のグラフ
            x = bisearch_max(-INF, INF, check2)
        # x = 0 ならカウントしない
        if abs(x) < EPS:
            pass
        elif x > 0:
            ans[0] += 1
        elif x < 0:
            ans[1] += 1
    # それ以外なら解は重解も含めると解は3つ
    else:
        ans1, ans2 = min(ans1, ans2), max(ans1, ans2)
        # /\/のグラフ
        if a > 0:
            res1 = bisearch_max(-INF, ans1, check1)
            res2 = bisearch_max(ans1, ans2, check2)
            res3 = bisearch_max(ans2, INF, check1)
        # \/\のグラフ
        else:
            res1 = bisearch_max(-INF, ans1, check2)
            res2 = bisearch_max(ans1, ans2, check1)
            res3 = bisearch_max(ans2, INF, check2)

        # 3つの解候補
        for x in [res1, res2, res3]:
            # f(x) = 0 について確認
            if abs(calc2(x)) < EPS:
                # x = 0 ならカウントしない
                if abs(x) < EPS:
                    continue
                elif x > 0:
                    ans[0] += 1
                elif x < 0:
                    ans[1] += 1
    print(*ans)
