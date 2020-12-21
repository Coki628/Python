# -*- coding: utf-8 -*-

"""
・自力AC
・インタラクティブ問題
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

# まずは素数で試して何の倍数か絞り込む
primes = eratosthenes_sieve(1000)
base = 0
for a in primes:
    print('? {0}'.format(a))
    sys.stdout.flush()
    yn = input()
    if yn == 'Y':
        base = a
        break
else:
    # 1の場合はどの素数でも割れない
    print('! {0}'.format(1))
    exit()

a = base * 2
NG = [0] * 1001
while a <= 1000:
    if not NG[a]:
        print('? {0}'.format(a))
        sys.stdout.flush()
        yn = input()
        if yn == 'Y':
            # YESなら今のaを基準に倍々するようにする
            base = a
        else:
            # NOなら今のaの倍数をチェック対象から外す
            b = a
            while b <= 1000:
                NG[b] = 1
                b += a
    a += base
print('! {0}'.format(base))
