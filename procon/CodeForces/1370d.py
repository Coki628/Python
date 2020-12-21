"""
・ここまで自力AC！
・最大値の最小化、二分探索
・偶奇の小さくする方を最大m以下に、の形でにぶたんする。これなら要素を貪欲に取りに行ける。
・なにげに条件整えるのはちょっと考えた。小さい方がKの半分集まっても、
　全体がK個集まらないと成立しないので、それも考慮して条件を作る。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def bisearch_min(mn, mx, func):
    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

# 偶奇で小さい方の最大値をm以下として、目的の数列を構築できるか
def check(m):
    # 奇数番目を小さくする
    cnt1 = 0
    cnt2 = 0
    i = 0
    while i < N:
        if A[i] <= m:
            cnt1 += 1
            i += 1
            if i < N:
                cnt2 += 1
                i += 1
        else:
            i += 1
    # 偶数番目を小さくする
    cnt3 = 0
    cnt4 = 1
    i = 1
    while i < N:
        if A[i] <= m:
            cnt3 += 1
            i += 1
            if i < N:
                cnt4 += 1
                i += 1
        else:
            i += 1
    # どちらかうまくいけばOK
    return (cnt1 >= ceil(K, 2) and cnt1+cnt2 >= K) or (cnt3 >= K // 2 and cnt3+cnt4 >= K)

N, K = MAP()
A = LIST()

res = bisearch_min(0, 10**9+1, check)
print(res)
