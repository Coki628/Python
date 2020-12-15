"""
・蟻本演習4-2-7
・山が分裂するGrundy数、メモ化再帰
・Grundy数は、途中で山が分裂しても使える。各遷移からの戻りをxorすればOK。
　集めてきた各遷移先のGrundy数に対する扱いは、
　Grundy数の定義通りmex取って現在状態のGrundy数を決める。
・つまり、1つの操作で山が分裂するところはxorするけど、別の操作での遷移は普通にGrundy数としてsetに集める。
　最初この辺の理解がモヤっとしてて、全部の戻り値xorしようとしてた。。
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

N, D = MAP()

memo = [-1] * (N+1)
def rec(n):
    if memo[n] != -1:
        return memo[n]
    se = set()
    for i in range(1, n):
        for j in range(i+1, n):
            k = n-i-j
            if k <= 0:
                break
            if i != j and j != k and k != i and max(i, j, k) - min(i, j, k) <= D:
                res = rec(i) ^ rec(j) ^ rec(k)
                se.add(res)
    g = 0
    while g in se:
        g += 1
    memo[n] = g
    return g

grundy = rec(N)
if grundy:
    print('kado')
else:
    print('matsu')
