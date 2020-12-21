"""
・自力AC
・累積和、式変形
・確かゆきこで類題を見た。(978)
　でも全ペアa*bの総和、ってのはありがちっぽいので他でも出てるはず。
・左端を決め打つと、右端を取りうるのはその先の全部で、この区間和は前計算しておけばO(1)で取れる。
・でもこれ灰diffはやばいと思うなー。
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

acc = [0] * (N+1)
for i in range(N):
    acc[i+1] = (acc[i]+A[i]) % MOD

ans = 0
for i in range(N-1):
    ans += A[i] * (acc[-1]-acc[i+1])
    ans %= MOD
print(ans)
