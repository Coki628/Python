"""
参考：https://www.hamayanhamayan.com/entry/2019/04/28/105452
・自力ならず。。全然方針立たなかった。
・数え上げ、nCr系
・ある数に注目して、その数が生き残る通り数を考えると、
　「ある数xが選ばれない通り数 + ある数xが選ばれてかつx+1がxより後に選ばれる通り数」となる。
・この2つはxが選ばれるか選ばれないかで背反になっているので、重複は発生せず普通に足せる。
・1つ目の条件はx以外のN-1個からK個を選んで好きな順序で並べるnPr(N-1, K)。
・2つ目の条件がちょっとややこしめだけど、まずK箇所からxとx+1の場所を決めるnCr(K, 2)、
　(なおその2つの順序は条件よりx+1が後と決まっている)
　続いて残りのK-2箇所に、残るN-2個からK-2個選んで好きな順序で並べるnPr(N-2, K-2)、となっている。
・式より、この通り数はxの値に寄らないので、先に1回計算して使い回せばいい。
・Nについてのみ、N+1は存在しないので2つ目の条件を除いて処理する。
・追記。ループ回したものの、中身は等差数列の和なのでO(1)でできる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

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

    def nHr(self, n, r):
        """ 重複組み合わせ """

        return self.nCr(r+n-1, r)

    def nPr(self, n, r):
        """ 順列 """

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

def get_sum(a, d, n):
    """ 等差数列の和：(初項a, 公差d, 項数n) """
    return (2*a + (n-1)*d) * n // 2

N, K = MAP()
mt = ModTools(N, MOD)

ans = (mt.nPr(N-1, K)*get_sum(1, 1, N) + mt.nCr(K, 2)*mt.nPr(N-2, K-2)*get_sum(1, 1, N-1)) % MOD
print(ans)
