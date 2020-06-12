"""
参考：ツイート等
・自力ならず。。
・素数列挙、約数、GCD
・わりと近いとこまで来てた。素数列挙してそれが候補、とかは間違ってなかった。
・最後bを小さくしないとと思ってgcd取っちゃったんだけど、
　素因数分解みたいに割れるだけ割るのが正解だった。。
・まあでもしっかり考察したってよりは雰囲気で進んだ感じだったから、いけなくてもしょうがない。
・hackされた。TLE。。。
"""

import sys
from math import sqrt, gcd

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

def eratosthenes_sieve(n):
    table = [0] * (n+1)
    prime_list = []
    for i in range(2, n+1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i+i, n+1, i):
                table[j] = 1
    return prime_list

N = INT()
A = LIST()
MAX = 10 ** 7

primes = eratosthenes_sieve(int(sqrt(MAX))+1)
S = set(primes)
ans1 = [0] * N
ans2 = [0] * N
for i in range(N):
    if A[i] in S:
        ans1[i] = ans2[i] = -1
        continue
    for prime in primes:
        if A[i] % prime == 0:
            b = A[i] // prime
            while b % prime == 0:
                b //= prime
            if b != 1 and gcd(prime+b, A[i]) == 1:
                ans1[i] = prime
                ans2[i] = b
                break
    else:
        ans1[i] = ans2[i] = -1
print(*ans1)
print(*ans2)
