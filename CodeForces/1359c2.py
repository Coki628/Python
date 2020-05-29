"""
・二分探索、誤差
・Decimal使わないで分数クラスFractionで処理した。
・これだと探索回数減らさなくても問題なく通った。pypyAC0.9秒。
・分数クラスになった後も普通にintと計算も比較もできるし誤差もないし、
　Decimalほど遅くならないから、有理数ならDecimalよりこっちのが誤差対策として有用かも。
"""

import sys
from fractions import Fraction

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
INF = 10 ** 10
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

# m*2+1回カップを入れた時、温度がt以上かどうか
def check(m):
    # 式変形で除算を避ける
    res = (h*(m+1) + c*m)
    return res >= t*(m*2+1)

for _ in range(INT()):
    h, c, t = MAP()

    res = bisearch_max(0, 10**10, check)
    if res == 10**10-1:
        print(2)
    else:
        mn = INF
        ans = -1
        # 誤差はないようで、前後2値確認でOK
        for m in range(res, res+2):
            # 分数クラスで小数を避ける
            res = Fraction(h*(m+1)+c*m, m*2+1)
            diff = abs(res - t)
            if diff < mn:
                mn = diff
                ans = m*2 + 1
        print(ans)
