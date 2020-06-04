"""
・自力AC
・素数列挙
・題意より、必要なのは「10^5より大きい素数のみで構成された合成数」(これが分かるまでに時間かかった。。)
・N<=10より、小さい方10個分かれば十分なので、3つペア以上を考える必要はなく、
　2つペアで小さい素数10個くらいを掛け合わせて全列挙、それをソートする。
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

ans = [1]
primes = eratosthenes_sieve(10**5*2)
A = []
for prime in primes:
    if prime > 10**5:
        A.append(prime)
A = A[:10]
B = [1]
for i in range(10):
    for j in range(i, 10):
        B.append(A[i]*A[j])
B.sort()
print(B[N-1])
