"""
・蟻本演習4-2-6
・Grundy数、素因数分解、約数列挙
・自力AC！やったね。まあ蟻本でGrundy数の取り方チェックしながらだけど。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = INT()
A = LIST()

grundy = [0] * N
for i in range(N):
    fact = factorize(A[i])
    divs = sorted(divisors(A[i]))
    dp = {}
    dp[1] = 0
    # A[i]の素因数を使って通る可能性がある値は約数divsだけ
    for div in divs[1:]:
        se = set()
        for k in fact:
            # 素因数kで割って行ける場所のGrundy数
            if div % k == 0:
                se.add(dp[div//k])
            # 素因数kで2回割って行ける場所のGrundy数
            if div % (k**2) == 0:
                se.add(dp[div//(k**2)])
        # mexを取ってここのGrundy数を求める
        g = 0
        while g in se:
            g += 1
        dp[div] = g
    grundy[i] = dp[A[i]]

ans = 0
for g in grundy:
    ans ^= g
if ans:
    print('Alice')
else:
    print('Bob')
