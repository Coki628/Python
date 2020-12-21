"""
参考：https://qiita.com/drken/items/3b4fdf0a78e7a138cd9a
・自力ならず。。
・合同式、式変形、MOD逆元(素数でない)、拡張ユークリッドの互除法
・TL見てしまっていたけど、とりあえず自力で式変形を考えてみる。
　x*K+S = 0 mod N から、x = -S/K mod N として、NとKが互いに素でないなら
　割り算できないのでNGとする。で、割り算はフェルマー、って思ったら何か答え合わない。。
・色々調べてみると、MODが素数じゃないとフェルマーでMOD割り算できないっぽい。
　言われてみたらそうだったね確かに。。でも互いに素なら何か行けなかったっけ？
　ってさらに調べると、けんちょんさんの記事に行き着く。ここで拡張ユークリッドの互除法が出てくる。
　やり方も書いてある。extgcd(逆元がほしい数, MOD) ってやった時のxが逆元。これを掛ければ無事、割り算。
・数学色強めに見えたし、実装頑張れば行けそうなFに張り付いてしまったけど、
　これならノーヒントで自力で調べても頑張れば行けたかもしれない。。うーん。。
"""

import sys
from math import gcd

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

def extgcd(a, b, x=1, y=0):
    """ 拡張ユークリッドの互除法(ax+by=gcd(a, b)の解を求める) """

    if b == 0:
        x = 1
        y = 0
        return (y, x)
    else:
        x, y = extgcd(b, a%b, y, x)
        y -= a // b * x
        return (y, x)

# MOD逆元で割り算(MODが素数でなくても、bとMODが互いに素なら可)
def div(a, b, MOD):
    _, x = extgcd(b, MOD)
    return a*x % MOD

for _ in range(INT()):
    N, S, K = MAP()

    g = gcd(gcd(N, S), K)
    N //= g
    S //= g
    K //= g

    if gcd(N, K) == 1:
        ans = div(-S, K, N)
        print(ans)
    else:
        print(-1)
