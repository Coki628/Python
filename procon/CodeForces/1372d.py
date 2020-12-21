"""
参考：https://twitter.com/laycrs/status/1281998883049009158
　　　https://codeforces.com/blog/entry/79974
・自力ならず。。。
・変形させてから累積和
・最終的に生き残ることになる値は元の配列の位置で交互になる。
・円環で、Nは奇数と決まっているので、最後まで行ったら偶奇逆を始めから、みたいな配列を作っておいて、
　累積和取って、Nの半分を区間で取る、の一番いいところが答え。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

A2 = []
for i, a in enumerate(A):
    if i % 2 == 0:
        A2.append(a)
for i, a in enumerate(A):
    if i % 2 == 1:
        A2.append(a)

K = ceil(N, 2)
A2 += A2
acc = [0] + list(accumulate(A2))
ans = 0
for i in range(N):
    ans = max(ans, acc[i+K] - acc[i])
print(ans)
