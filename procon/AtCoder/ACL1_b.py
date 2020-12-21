"""
参考：https://twitter.com/long_ng_cp/status/1307689296820797440
・むずかった。こっちはいくら時間あっても自力いけなかったと思う。
・等差数列の和、素因数分解、ビット全探索、拡張ユークリッドの互除法
・まず、初項1,公差1,項数kの等差数列の和を式変形するとして、k*(k+1)= 2Nの倍数を目指す。
　(ここからして解法聞いてからピンと来るまでに時間かかった。数学力さん。。。)
・kとk+1は互いに素なので、素因数を種類毎にA,Bと振り分けると、ビット全探索で全通りが試せる。
・拡張ユークリッドの互除法を使うと、ax+by=gcd(a, b)の解が求まる。これを使う。
　今回はa,bは互いに素なのでGCDは1、ax+by=1を求めることができる。
　本当に欲しいのはabs(ax-by)=1だけど、ここは係数x,yの正負を調整すればうまく作れる。
・答えとして欲しいのは項数kなので、kとk+1に当たるabs(ax)とabs(by)の小さい方。
・ただ、これを普通にやると、片方を1つも使わずに0,1にしてこれで差は1だねー、
　も答えとして使われてしまうので、xかyが0のやつが来たら弾く。
・ちょっと罠なのは、答えの初期値をINFにしておくと、2冪が来た時に答えを使えるやつが1つもなくてダメになる。
　(他の冪乗は、図らずも今回2Nにするために*2しているおかげで素因数が1つにならず、助かる。)
　なので初期値はNにしておく。(NならばN*(N+1)/2とできるので絶対使える。)
・なかなかWAが取れなかったのは、冪乗するとこを*にしてた単なるケアレスミスでした。。。
・素因数分解の√Nが今回10^15でちょっと間に合うか心配になったけど、問題なくpypyAC0.54秒。
・ビット全探索パートは素因数が最大で13個くらいしかないそうなので大丈夫。
　内側でGCDのlogが乗るけど、それでもまあ√Nよりは少ないでしょう多分。
・TODO: 想定解はCRT(中国剰余定理)を用いたものなので、時間ある時にそっちでも解いてみよう。
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

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

def extgcd(a, b, x, y):
    """ 拡張ユークリッドの互除法(ax+by=gcd(a, b)の解を求める) """

    if b == 0:
        x = 1
        y = 0
        return (y, x)
    else:
        x, y = extgcd(b, a%b, y, x)
        y -= a // b * x
        return (y, x)

N = INT()

fact = list(factorize(N*2).items())
M = len(fact)

ans = N
for bit in range(1<<M):
    a = b = 1
    for i in range(M):
        if (bit>>i) & 1:
            a *= fact[i][0] ** fact[i][1]
        else:
            b *= fact[i][0] ** fact[i][1]
    y, x = extgcd(a, b, 0, 0)
    if x == 0 or y == 0:
        continue
    res = min(abs(a*x), abs(b*y))
    ans = min(ans, res)
print(ans)
