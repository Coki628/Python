"""
参考：https://twitter.com/kyopro_friends/status/1274705953099669505
　　　https://www.youtube.com/watch?v=TUdZT1wIbe8&feature=youtu.be
・自力ならず。。
・数え上げ
・重複なく数えるために工夫がいる。
・ある固定の文字をどの位置に入れるかを考えることになるが、この時、aが最初に出た位置、
　a以降でbが最初に出た位置、b以降でc...みたいに前から貪欲に決め打つと重複を防げる。
・このようにするために、aが出る位置より手前にはaを置かない、a以降bが出る位置より手前には
　bを置かない、と考えると、結局最後の文字を置く位置までは全部26-1個のアルファベットのどれかを
　置くことになる。これが分かると、全体のうち、最後の固定の文字より前は25通り、それ以降は26通りから
　自由に選べるとすれば最後の固定の文字の位置でループ回して解ける。
・固定の文字の各配置をどうするか、は普通のnCrで出せる。
・600点級の数え上げ、なかなか自力ACできないなぁ。頑張って理解して経験値積もうー。
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

K = INT()
S = input()
N = len(S)

mt = ModTools(2*10**6, MOD)
ans = 0
for x in range(K+1):
    ans += mt.nCr(N+K-x-1, N-1) * pow(25, K-x, MOD) * pow(26, x, MOD) % MOD
ans %= MOD
print(ans)
