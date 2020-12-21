"""
参考：https://yukicoder.me/problems/no/978/editorial
・公式解の方針。こっちでもやっておく。
・Ai*Ajを2次元平面上の領域の面積と考えると、
　全体のちょうど半分から、真ん中の部分だけからもう半分増やした状態が答えになる。
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N, p = MAP()

if N == 1:
    print(0)
    exit()

A = [0] * N
A[1] = 1
for i in range(2, N):
    A[i] = (A[i-1]*p + A[i-2]) % MOD

allsm = pow(sum(A), 2, MOD)
centersm = 0
for i in range(N):
    centersm += pow(A[i], 2)
    centersm %= MOD
ans = (fermat(allsm, 2, MOD) + fermat(centersm, 2, MOD)) % MOD
print(ans)
