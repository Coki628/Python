"""
・自力AC
・素因数分解、約数列挙
・ちょっと条件がややこしい。でも多分本質は、素因数全部にしても、その後値を増やしていくにしても、
　結局あんまり回数が多くならないから全探索いけるよってことかなと思った。
・多分、数学考察強いともっと頭良くやれるやつ。
"""

import sys
from itertools import combinations

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
EPS = 10 ** -10

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

N, K = MAP()

fact = factorize(N)
A = []
for k, v in fact.items():
    A += [k] * v

M = len(A)
ans = mx = 0
# Nの全素因数からK個選ぶ組み合わせを全列挙
for comb in combinations(A, K):
    num = 1
    for c in comb:
        num *= c
    # 元になる素因数は維持しつつ、N未満の間、値を大きくする
    cur = num
    while cur < N:
        # 各値で約数の個数を出す
        ln = len(divisors(cur))
        if ln > mx:
            mx = ln
            ans = cur
        elif ln == mx:
            if cur < ans:
                ans = cur
        cur += num
print(ans)
