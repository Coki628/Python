"""
・自力AC
・周期、GCD系
・とりあえずいくつか試していって、N,Dが互いに素だと全部埋まるまでぶつからなそうだなー、となる。
・じゃあそうじゃない時は、、となると、互いに素の時に/1でいけるなら、そうじゃない時もGCDで合う
　んじゃないか、っていう読み。実際にやってみると合う。証明じゃない、経験則。
　よくないよねー、ちゃんと証明したいよねー、でも結局色々試して経験則からそれっぽいとこに行ってしまう。
・あと踏んだ足場の数じゃなくて移動可能回数だから、-1するね。
"""

import sys
from math import gcd

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

N, D = MAP()

g = gcd(N, D)
ans = N // g - 1
print(ans)
