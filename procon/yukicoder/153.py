"""
・蟻本演習4-2-7
・山が分裂するGrundy数、メモ化再帰
・やり方は覚えたのでこれは自力AC！
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

N = INT()

memo = [-1] * (N+1)
def rec(n):
    if memo[n] != -1:
        return memo[n]
    se = set()
    if n >= 2 and n % 2 == 0:
        se.add(rec(n//2) ^ rec(n//2))
    if n >= 3 and n % 2 == 1:
        se.add(rec(n//2) ^ rec(n//2+1))
    if n >= 3 and n % 3 == 0:
        se.add(rec(n//3) ^ rec(n//3) ^ rec(n//3))
    if n >= 4 and n % 3 == 1:
        se.add(rec(n//3) ^ rec(n//3) ^ rec(n//3+1))
    if n >= 5 and n % 3 == 2:
        se.add(rec(n//3) ^ rec(n//3+1) ^ rec(n//3+1))
    
    g = 0
    while g in se:
        g += 1
    memo[n] = g
    return g

ans = rec(N)
if ans:
    print('A')
else:
    print('B')
