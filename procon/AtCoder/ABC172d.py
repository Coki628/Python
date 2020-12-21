"""
・ここまで自力AC
・素数列挙応用、約数の個数
・各約数の個数はエラトステネスのループで倍数に足しこんで行けばNlogNではいける。
・でも今回制約が1000万だったので、これ想定違くないか？ってなった。
・C++で書き直すか迷ったけど、順位表の通され方と、新環境pypyの速度を信じて提出。無事pypyAC1.9秒。(制約3秒)
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
    table = [0] * (n+1)
    for i in range(1, n+1):
        for j in range(i, n+1, i):
            table[j] += 1
    return table

N = INT()

res = eratosthenes_sieve(N)
ans = 0
for i in range(1, N+1):
    ans += res[i] * i
print(ans)
