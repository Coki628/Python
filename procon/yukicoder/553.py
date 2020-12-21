"""
参考：https://juken-mikata.net/how-to/mathematics/inverse-function.html
・自力AC！
・逆関数、極値
・逆関数の求め方はググった。冪乗の逆はlogっぽいなーと思いつつも自分でうまく作れなかったけど、
　ほぼ同じやつが上記ページの例に載っててよかった。(2冪なら両辺にlog2する)
・F(INF)は、適当に大きい値でF(n)を試すと、ほぼある値に収束していたので、それを固定値として
　作っておいて使った。
・シグマのとことか、pythonだとリストのsumで直感的に同じことができるのでいい感じ。
"""

import sys
from math import sqrt, log2

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def F(n):
    return sqrt(sum(0.81**i for i in range(1, n+1))) / sum(0.9**i for i in range(1, n+1))
# F(INF)
FINF = 0.22941573387

def f(n):
    return (F(n) - FINF) / (F(1) - FINF) * 1200

def g(X):
    return 2**(X/800)

# gの逆関数
def grev(X):
    return 800 * log2(X)

def calc(A):
    n = len(A)
    return grev(sum(g(A[i])*0.9**(i+1) for i in range(n)) / sum(0.9**i for i in range(1, n+1))) - f(n)

N = INT()
A = LIST(N)
ans = int(calc(A))
print(ans)
