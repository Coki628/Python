"""
・自力AC！しっかり考察できてよかった。
・BIT、転倒数応用、真ん中固定
・3つ組を求めるやつなので、とりあえず真ん中固定の方針で考えてみる。
　どういう数が足されるか、うまくまとめられないか色々考えていると、
　「左の総和*右の個数 + 左の個数*右の総和 + 自身*左の個数*右の個数」
　ってやつをやったら合いそうだな、ってなる。
・BIT2本持って個数と総和を両方管理、みたいのはどっかのこどふぉで見たことある。
・なんか余事象使えば片側だけ持ってもいけそうだけど、思考リソースを割きたくなかったので
　贅沢に左右で分けて、計4本のBIT生やしてやった。
・制約を見るとmax(A)<=10^9なので座圧がいるな、ってなる。
・あとはBITに通り過ぎたら詰めたり抜いたりしながら、左右の数を適切に管理して数えられればOK。
・20万に座圧BITは重めかなーとは思ったけど、無事pypyAC1.10秒。
"""

import sys

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
MOD = 998244353
EPS = 10**-10

class BIT:
    """ Binary Indexed Tree (MOD付) """

    def __init__(self, n):
        self.n = n
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            s %= MOD
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            self.tree[i-1] %= MOD
            i += i & -i

    def query(self, l, r):
        """ 区間和の取得 [l, r) """
        return (self.sum(r-1) - self.sum(l-1)) % MOD
    
    def get(self, i):
        """ 一点取得 """
        return self.query(i, i+1)

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
A = LIST()

zp, unzp = compress(set(A))

MX = len(zp)
bitcntl = BIT(MX)
bitcntr = BIT(MX)
bitsml = BIT(MX)
bitsmr = BIT(MX)
for i in range(N):
    bitcntr.add(zp[A[i]], 1)
    bitsmr.add(zp[A[i]], A[i])

ans = 0
for i in range(N):
    cntl = bitcntl.query(zp[A[i]]+1, MX)
    cntr = bitcntr.query(0, zp[A[i]])
    sml = bitsml.query(zp[A[i]]+1, MX)
    smr = bitsmr.query(0, zp[A[i]])
    # 左の総和*右の個数 + 左の個数*右の総和 + 自身*左の個数*右の個数
    ans += sml*cntr + cntl*smr + A[i]*(cntl*cntr)
    ans %= MOD
    bitcntl.add(zp[A[i]], 1)
    bitcntr.add(zp[A[i]], -1)
    bitsml.add(zp[A[i]], A[i])
    bitsmr.add(zp[A[i]], -A[i])
print(ans)
