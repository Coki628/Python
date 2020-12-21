"""
・自力AC！水diff上位の数学問倒した。
・wolframalpha
・前回うまくでなかったやつをちょっと変えて、
　sum(comb(N, i*2) * M**(i*2), for i=0 to N)
　sum(comb(N, i*2+1) * M**(i*2+1), for i=0 to N)
　としたら、答えにそのまま繋がる式が出た。最初、これだとi=2*Nくらいまで数えちゃうから
　N半分にしないとダメじゃないかとか思ったんだけど、
　i>Nになった以降はnCr(N, i)が0になっちゃうから、その先は気にしなくていいんだね。
　実際、i=0 to N/2 ってしても、同じ式が出てくる。
・まず愚直ループのシグマの式立てて、ちゃんと適切に式変形して、
　wolframalphaが解釈しやすい形にして食わせる、っていうイメージしてた形をしっかり作れたからOK。
　公式解説の説明はよく分からん。。。
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

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版 """
    from math import factorial

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

def div(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

for _ in range(INT()):
    N, M, X = MAP()

    # res = 0
    # for i in range(N+1):
    #     res += nCr(N, (i*2+1)) * M**(i*2+1)
    #     res %= MOD
    if X:
        # res = ((M+1)**N - (1-M)**N) // 2
        res = div(pow(M+1, N, MOD) - pow(1-M, N, MOD), 2, MOD)
    else:
        # res = ((1-M)**N + (M+1)**N) // 2
        res = div(pow(1-M, N, MOD) + pow(M+1, N, MOD), 2, MOD)
    # res %= MOD
    print(res)
