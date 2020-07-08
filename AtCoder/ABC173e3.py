"""
・誤差対策
・numpyでlongdouble使ったら通ったー。。
"""

import sys
import numpy as np

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

N, K = MAP()
A = LIST()

A1 = []
A2 = []
zero = 0
for a in A:
    if a > 0:
        A1.append(a)
    elif a < 0:
        A2.append(-a)
    else:
        zero += 1

B1 = np.log(np.array(A1, dtype=np.longdouble))
B2 = np.log(np.array(A2, dtype=np.longdouble))
A1.sort()
A2.sort()
B1.sort()
B2.sort()
acc1 = np.insert(B1.cumsum(), 0, 0)
accrev1 = np.insert(B1[::-1].cumsum(), 0, 0)
acc2 = np.insert(B2.cumsum(), 0, 0)
accrev2 = np.insert(B2[::-1].cumsum(), 0, 0)
mx = -np.inf
mn = np.inf
idx1 = (-1, -1)
idx2 = (-1, -1)
for i in range(K+1):
    j = K - i
    if len(B1) < i or len(B2) < j:
        continue
    if j % 2 == 0:
        sm = accrev1[i] + accrev2[j]
        if sm > mx:
            mx = sm
            idx1 = (i, j)
    else:
        sm = acc1[i] + acc2[j]
        if sm < mn:
            mn = sm
            idx2 = (i, j)

if mx != -np.inf:
    ans = 1
    n, m = idx1
    A1.reverse()
    A2.reverse()
    for i in range(n):
        ans *= A1[i]
        ans %= MOD
    for i in range(m):
        ans *= -A2[i]
        ans %= MOD
    print(ans)
elif zero:
    print(0)
else:
    ans = 1
    n, m = idx2
    for i in range(n):
        ans *= A1[i]
        ans %= MOD
    for i in range(m):
        ans *= -A2[i]
        ans %= MOD
    print(ans)
