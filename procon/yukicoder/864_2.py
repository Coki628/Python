"""
参考：https://drive.google.com/file/d/1x_2jN_sumw2ZJAuxiZiMS-RUCqZYprOa/view
・自力ならず。。
・約数列挙、式変形
・式変形が本質。全然この方針に行けなかった。。
　ab + bc + cd + da は式変形すると、(a+c) * (b+d) にできる。
・あとは約数列挙して、全約数について対になるやつとのペアで確認。
・N以下でx=a+cを作れるかは、やってみると、最大はx-1個で、
　Nがx-1より小さくなる毎に2つずつ減っていくような形になると分かる。
・N<=10^12で素因数分解か約数列挙は疑ったんだけどなー。全然式変形する頭がなかった。。
　うまいことnCrで数え上げられないかとか、全然ダメ方針で考えてしまった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = INT()
K = INT()

def check(x):
    return x-1 - max(min(2*(x-1-N), x-1), 0)

divs = sorted(divisors(K))
M = len(divs)
ans = 0
for i in range(M):
    a = divs[i]
    b = K // divs[i]
    ans += check(a) * check(b)
print(ans)
