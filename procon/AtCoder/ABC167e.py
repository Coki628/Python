"""
参考：https://www.youtube.com/watch?v=ENSOy8u9K9I&feature=youtu.be
　　　https://twitter.com/kyopro_friends/status/1259483253120987138
・自力ならず。。すごくあっとこにありがちな系なのにね。。
・数え上げ
・隣り合う場所がk箇所、を0〜Kまでやる、の方針は合ってた。
・色を同じにする要素を考えた時に、例えばk=2ならooxxooとoooxxxみたいに、
　配置によって同じ色にする要素数が変わってしまう。これ何個連続してるかとかで
　場合分けするん？いや、無理だろそんなの。。で詰んで終了。。。
・正解は、グループでまとめる。最初、そう聞いても、例えば上の例ならOxxOとOxxxみたいに
　(あるいはOXOとOX？)結局状態違くなってダメじゃん、って思ってしまったんだけど、
　そうではなくて、「グループ数」がいくつになるかで考える。同じ値にする要素のグループと
　そのままのグループを分けて考える必要がない。単純にOxxOとOxxxなら両方グループ4つだね、でいい。
　すると各グループに対する色の割り当て方は、k=0の時と何も変わらない。数が減ってるだけ。
・それともう1つ、「どの位置を連結させるか」は数える必要がある。
　これは全要素の隣り合う場所N-1個からk個選ぶnCrでいける。
・ここまで整うと、後はループ回してMOD取って数えるだけ。あっとこド定番の数え上げ。
　いっぱいやってるのに、考察できなくて解けないの悲しい。。。
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
MOD = 998244353

class ModTools:
    """ 階乗たくさん使う時用のテーブル準備 """

    def __init__(self, MAX, MOD):
        """ MAX：階乗に使う数値の最大以上まで作る """
        
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        inverse = [1] * MAX
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse
    
    def nCr(self, n, r):
        """ 組み合わせの数 """

        if n < r: return 0
        r = min(r, n-r)
        numerator = self.fact[n]
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

N, M, K = MAP()

mt = ModTools(N, MOD)
ans = 0
for k in range(K+1):
    ans += M * pow(M-1, N-k-1, MOD) * mt.nCr(N-1, k)
    ans %= MOD
print(ans)
