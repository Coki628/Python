"""
・結構すんなり自力AC
・素数列挙、ゲームDP、メモ化再帰
・Nが1万、再帰の各回で素数最大1000個以上のループが回るので、計算量心配だったけど無事pythonAC0.5秒。
・再帰の計算量は読めんなー。一応、勝ち確定した時点でループを打ち切ってるから枝刈りになってるとは思うんだけど。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

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
primes = eratosthenes_sieve(N)

memo = [-1] * (N+1)
def rec(n):
    if memo[n] != -1:
        return memo[n]
    if n <= 1:
        memo[n] = 1
        return 1
    for prime in primes:
        if prime > n:
            break
        if not rec(n-prime):
            memo[n] = 1
            return 1
    memo[n] = 0
    return 0

res = rec(N)
if res:
    print('Win')
else:
    print('Lose')
