"""
・自力AC
・素数列挙
・基本的には素数列挙さえできてれば、後は各素数についてO(1)で通り数を出せる。
・がしかし制約が1秒で、最初pypyTLEした。
・さらにちょっと考察して、鴨が最低3羽いるなら間隔は最大でもL/2くらいまでにしかならない。
　ならLまでの素数はいらないな、となってL/2の素数列挙にしたらAC0.5秒。
・でもこれC++ならLまでの素数列挙でも通りそうだよなー、1段階考察飛ばせるのはやっぱり羨ましい。
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

N, L = MAP()

primes = eratosthenes_sieve(L//2)

ans = 0
for a in primes:
    ln = a * (N-1)
    if ln > L:
        break
    ans += L - ln + 1
print(ans)
