"""
・さくっと自力AC！
・ソートして累積和
・いわゆるペアの和を求める系なんだけど、絶対値なのが話をややこしくしてる。
　ここで先にソートしておくと、自分より先が全部自分より大きくなるので、
　自分を左端として、右端は自分より右の総和-自分*まとめる区間数、みたいにできる。
・ちょっと考察も要る系の累積和問題だと思うんだけど、これで茶diff前半ってやばい。
"""

import sys
from itertools import accumulate

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
sys.setrecursionlimit(10**6)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = LIST()

A.sort()
acc = list(accumulate(A[::-1]))[::-1]

ans = 0
for i in range(N-1):
    ans += acc[i+1] - A[i]*(N-i-1)
print(ans)
