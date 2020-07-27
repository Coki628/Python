"""
・ここまで自力AC
・貪欲
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

N = INT()
A = LIST()

up = False
sell = [0] * N
buy = [0] * N
for i in range(N-1):
    if not up and A[i] < A[i+1]:
        up = True
        buy[i] = 1
    if up and A[i] > A[i+1]:
        up = False
        sell[i] = 1
sell[N-1] = 1

money = 1000
stock = 0
for i in range(N):
    if buy[i]:
        d, m = divmod(money, A[i])
        stock += d
        money = m
    if sell[i]:
        money += stock * A[i]
        stock = 0
print(money)
