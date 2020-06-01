"""
・自力AC
・範囲内どの値を使ってもいいので、最終的に最小と最大の間の値は全て取りうる、ってやつ。
　偶奇で場合分けする。偶数の方は.5も全部取るので2倍。
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

N = INT()
A = [0] * N
B = [0] * N
for i in range(N):
    A[i], B[i] = MAP()

A.sort()
B.sort()
if N % 2 == 0:
    mn = (A[N//2-1]+A[N//2]) / 2
    mx = (B[N//2-1]+B[N//2]) / 2
    ans = int((mx-mn)*2) + 1
else:
    mn = A[N//2]
    mx = B[N//2]
    ans = (mx-mn) + 1
print(ans)
