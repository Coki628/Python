"""
参考：https://kmjp.hatenablog.jp/entry/2014/12/15/0930
・蟻本演習4-2-6
・Grundy数、素数列挙
・自力で解いたやつは、各回でGrundy数を都度求めたけど、先に素数列挙しておけば、
　最大値以下のGrundy数を前計算で全部求めておいて、あとはそれを見て数列Aを確認するだけ、ってできる。
・1万までのループの内側で、1万までの素数のループを回すけど間に合う。pypyAC0.50秒。
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

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

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

# 使う最大値MまでのGrundy数を前計算
M = max(A)
primes = eratosthenes_sieve(M)
grundy = [0] * (M+1)
grundy[1] = 0
for x in range(2, M+1):
    se = set()
    for p in primes:
        if x % p == 0:
            se.add(grundy[x//p])
        if x % p**2 == 0:
            se.add(grundy[x//p**2])
    g = 0
    while g in se:
        g += 1
    grundy[x] = g

nim = 0
for a in A:
    nim ^= grundy[a]
if nim:
    print('Alice')
else:
    print('Bob')
