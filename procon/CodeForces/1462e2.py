"""
参考：https://twitter.com/momohara_kyopro/status/1338885650787966977
・自力ならず。。
・nCr数え上げ、ソート、先頭固定、二分探索
・自分ではCounterして状態をまとめようとしてたけど、それが却って良くなくて、
　ソートだけして値はまとめない方がよかった。。これなら、ある1つの値は固有のものになるので、
　そこを固定して重複なく数え上げができる。こういう先頭固定して、自分より後ろだけ考えることで
　重複なく数え上げるみたいなやつ、ちょくちょく出てくるよな。ものにしたい。。
・で、先頭要素を決め打つと、そこからK離れたところにあるやつだけが一緒に組めて、
　それはにぶたんですぐ求められて、nCrでそこからM-1(自分を除いた数)を選び出す、ってできる。
・んで次の要素を軸にする時には、そこより手前は対象に含めないので、重複なく数え上げできるんだよな。
"""

import sys
from bisect import bisect_right

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):

        # nCrならn、nHrならn+rまで作る
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
        """ 組み合わせ """

        if n < r: return 0
        r = min(r, n-r)
        numerator = self.fact[n]
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

for _ in range(INT()):
    N, M, K = MAP()
    A = LIST()

    mt = ModTools(N, MOD)
    A.sort()
    ans = 0
    for i in range(N):
        j = bisect_right(A, A[i]+K, lo=i) - 1
        ans += mt.nCr(j-i, M-1)
        ans %= MOD
    print(ans)
