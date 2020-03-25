# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=zbjSIZ16x_I&feature=youtu.be
　　　https://img.atcoder.jp/agc043/editorial.pdf
・ルーカスの定理(二項係数nCkの偶奇)
・絶対値が欲しいので、最初に全部-1して1,2,3→0,1,2としていい。
・まず、答えの偶奇がどうなるかを考える。
　すると、隣り合う値同士の絶対値の計算はxorと同じとみなせる。
　xorであれば、元々奇数1だったところが最終的にそれぞれ何回使われるかが分かれば、その偶奇が答えになる。
・何回使われるかは、答えに着くまでに、左右にそれぞれ何回行くのかで、
　N-1回からi回の左(N-1-i回の右でもいい)を選ぶ組み合わせの数、と考えられる。
　(ちょうど右と下しか行けないグリッドの経路数数える時みたいな感じ)
・このnCkの偶奇自体は知っていればO(1)で求められる。
・これで答えが1の場合は確定。
・2の場合について、まず1が1つでもあると2にはならない。
　これは証明もできるんだろうけど、実験してエスパーするのが現実的っぽい。
・これで残っているのは0か2しか含まれないものだけとなる。
　それらは0と2の間で、ちょうど偶奇でやった時の0と1の間で起こったのと同じことが起こるので、
　2を1とみなして、最初にやったのと同じ偶奇チェックをやればいい。
・考察重いなー。。これでもまだ青diffかと思うと、みんなすごいな、としか言えない。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def lucas(n, k):
    """ ルーカスの定理(二項係数nCkの偶奇を返す) """
    return (n&k) == k

N = INT()
A = [a-1 for a in list(map(int, input()))]

def check(A):
    mod2 = 0
    for i, a in enumerate(A):
        if a == 1:
            mod2 += lucas(N-1, i)
            mod2 %= 2
    return mod2

res = check(A)
if res == 1:
    print(1)
    exit()

if A.count(1):
    print(0)
    exit()

A = [a//2 for a in A]
res = check(A)
if res == 1:
    print(2)
else:
    print(0)
