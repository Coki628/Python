"""
・さくっと自力AC
・GCD系、約数の性質
・やるだけか、と思ったら違った。陽に約数列挙やろうとすると、余程速いやつ使わない限り10^18はきつい。
・どういう時約数の個数が偶数、奇数になるかを考えると、Nの約数は基本的にa*b=Nで対になったみたいな
　形をしていることを思い出す。つまりだいたいは偶数で、奇数なのはa=bみたいな約数がある時だけ。
　a*a=a^2で表せるかどうかが論点。これだ、と。
・floatのis_integerで誤差大丈夫かちょっと心配だったけど、今回は普通に通った。
"""

import sys
from math import gcd, sqrt

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

a, b = MAP()

g = gcd(a, b)

if sqrt(g).is_integer():
    print('Odd')
else:
    print('Even')
