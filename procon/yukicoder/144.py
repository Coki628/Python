"""
・自力AC
・素数列挙応用、期待値
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

def eratosthenes_sieve(n, p):
    # exp[i] := iが素数として消されずに生き残る確率
    exp = [1] * (n+1)
    for i in range(2, n+1):
        for j in range(i+i, n+1, i):
            # 処理が行われない(この数が消されずに生き残る)確率を掛ける
            exp[j] *= (1-p)
    return exp

N, p = input().split()
N = int(N)
p = float(p)

res = eratosthenes_sieve(N, p)
ans = 0
for i in range(2, N+1):
    ans += res[i]
print(ans)
