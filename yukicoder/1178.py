"""
参考：http://w3e.kanazawa-it.ac.jp/math/category/kansuu/iroiro-kansu/henkan-tex.cgi?target=/math/category/kansuu/iroiro-kansu/ennohouteisiki.html
・自力AC
・円の方程式
・半径を求めたいので、とりあえず (x-a)^2+(y-b)^2=r^2 の形に持っていくことを考える。
・式変形こねくり回す。(自分ノート参照)
・プログラムには変形した式の右辺書くだけ。
"""

import sys
from math import sqrt

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

a, b, c, d, e, f = MAP()

ans = sqrt(f/a-e/a+(c/(2*a))**2+(d/(2*a))**2)
print(ans)
