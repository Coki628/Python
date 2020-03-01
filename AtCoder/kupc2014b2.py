# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/B.7b93ab85.pdf
・公式解。さすがに解き方が的確。
・素因数分解の逆みたいな感じで、ある素数が何個含まれるかを調べていく。
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

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

primes = eratosthenes_sieve(1000)
ans = 1
for a in primes:
    print('? {0}'.format(a))
    sys.stdout.flush()
    yn = input()
    if yn == 'Y':
        i = 1
        while a**(i+1) <= 1000:
            print('? {0}'.format(a**(i+1)))
            sys.stdout.flush()
            yn = input()
            if yn == 'Y':
                i += 1
            else:
                break
        ans *= a**i
print('! {0}'.format(ans))
