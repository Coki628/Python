"""
・さくっと自力AC！
・XOR、区間
・左を固定して、自分より右で自分とペアになれるような数を考えた時、x ^ y = X は
　y = x ^ X なので、自分より右にあるyの数を数える。
　予めCounterしておけば、これはO(1)で分かるし、自分の番が来た段階でこのCounterから自分を引く
　ようにしていけば、うまく自分より右側だけから数を取れる。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

N, X = MAP()
A = LIST(N)

C = Counter(A)
ans = 0
for x in A:
    C[x] -= 1
    y = x ^ X
    ans += C[y]
print(ans)
