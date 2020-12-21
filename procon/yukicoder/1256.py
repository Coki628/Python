"""
・色々遠回りしてAC。。
・とりあえず長さ3以上の題意の数列を順番に見ていくと、長さ3なら公差3の等差数列みたいになっていく。
　なので長さ固定してmod 3で割り切れるか、みたいなのを適当なとこまで見ていくかーってやったら微妙にWA。。
・上限上げたりしても変化がないので問題をよく読むと、整数の、とあるので、これ負数でもいいのかってなって、
　それを踏まえると一気に構築可っぽいものが増えて、さらにWAを叩きながらも考えると、
　結局のところこれ1以外何でも作れるんじゃあ、、ってなってはい無事AC。
・こういうのでモタつくからこどふぉの前半でいつもやられるんだよな。。。
"""

import sys

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

if N == 1:
    NO()
else:
    YES()
