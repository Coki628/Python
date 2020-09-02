"""
参考：https://yukicoder.me/problems/no/731/editorial
　　　https://ja.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%97%E6%B3%95#%E4%B8%80%E6%AC%A1%E6%96%B9%E7%A8%8B%E5%BC%8F%E3%81%AE%E5%A0%B4%E5%90%88
・自力ならず。これは知識ゲーでしょう、しゃあないかな。
・最小二乗法
・そういうのがある。1次関数なら式に当てはめれば一発で出る。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = LIST()

# 最小二乗法
xsm = ysm = xysm = x2sm = 0
for x, y in enumerate(A):
    xsm += x
    ysm += y
    xysm += x * y
    x2sm += x**2
d = (N*xysm - xsm*ysm) / (N*x2sm - xsm**2)
b = (x2sm*ysm - xysm*xsm) / (N*x2sm - xsm**2)

cost = 0
for i in range(N):
    cost += (A[i] - (b+d*i))**2
print(b, d)
print(cost)

