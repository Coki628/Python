"""
参考：https://yukicoder.me/problems/no/811/editorial
・公式解
・a,b共通の素因数の個数はgcd(a, b)の素因数の個数となる。
・実は制約10万だとNまでの全ての数に素因数分解と約数列挙をやっても間に合う。
　(言われると確かにN√Nだからそうなんだよな。。)
"""

import sys
from math import gcd

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

ans = mx = 0
# 1からN-1までの数を全探索
for i in range(1, N):
    g = gcd(i, N)
    # i,N共通の素因数の個数を求める
    cnt = sum(factorize(g).values())
    # K個以上あれば、答え候補として約数の個数を確認
    if cnt >= K:
        ln = len(divisors(i))
        if ln > mx:
            mx = ln
            ans = i
        elif ln == mx:
            if i < ans:
                ans = i
print(ans)
