"""
・自力AC！
・工夫して全探索
・実験すると、1,2,3のどれかになるみたいだ、となる。
・であれば、1個で作れるか、2個で作れるかだけチェックして、無理なら3個、で行けそう。
・実際、これでAC。しかも想定解法だった。
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

A = [0]
for i in range(1, N+1):
    A.append(A[-1]+i)
    if A[-1] > N:
        break

M = len(A)
if N in A:
    print(1)
    exit()
for a in A:
    for b in A:
        if a+b == N:
            print(2)
            exit()
print(3)
