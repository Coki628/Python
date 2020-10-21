"""
・自力AC！
・素因数分解、メモ化再帰
・とりあえず考察すると、素因数を3つに分けた時のx + y + zが最小になるような組が欲しい、となる。
・で、最大ケースで素因数の数を見ると22個、案外少ない。けど3^N全探索をやるにはつらい。
・ソートしたりプリキューに詰めたりで貪欲しても無理だったし、他に妙案も出ないのでとりあえずやってみる。
・一応、素因数は2とかかなり同じ数も入っているので、絶対途中でも重複が結構あるだろう、とは思った。
　ので、状態4つもあって微妙だけど、メモ化再帰してみる。通った。pypyAC2.66秒。(制約5妙)
・試しにメモなしで再帰全探索投げたらTLEしたので、まあメモ化がちゃんと効いているのが分かった。
・公式より、確かに順番は関係なかったな、と思って、ソートしてからメモ化やるようにしたら、
　かなり速くなってpypyAC0.56秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**18
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

N = INT()

fact = factorize(N)
A = []
for k, v in fact.items():
    A += [k] * v
A.sort(reverse=1)
M = len(A)

memo = {}
def rec(x, y, z, i):
    if i == M:
        return x + y + z - 3
    x, y, z = sorted([x, y, z])
    if (x, y, z, i) in memo:
        return memo[(x, y, z, i)]
    res = min(
        rec(x*A[i], y, z, i+1),
        rec(x, y*A[i], z, i+1),
        rec(x, y, z*A[i], i+1),
    )
    memo[(x, y, z, i)] = res
    return res

mn = rec(1, 1, 1, 0)
print(mn, N-1)
