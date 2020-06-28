"""
・数え上げ、包除原理
・ABC172eで包除やったのでこれを復習してみる。考え方はやっぱり写像3相と同じ。
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
MOD = 1777777777
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

def nCr(n, r, MOD):
    """ 組み合わせの数(大きいnに対して使用する。計算量はr) """

    if n < r: return 0
    r = min(r, n-r)
    num = den = 1
    for i in range(n, n-r, -1):
        num *= i % MOD
        num %= MOD
    for i in range(r, 0, -1):
        den *= i % MOD
        den %= MOD
    return num * pow(den, MOD-2, MOD) % MOD

N, K = MAP()

# 自分以外の手紙が届く所を数える
mt = ModTools(K+1, MOD)
ans = 0
for i in range(K+1):
    # 確実に自分以外の手紙が届くi個をK個から選ぶ * 残りのK-i箇所を制限なく並べる * 包除で+-
    ans += mt.nCr(K, i) * mt.fact[K-i] * (-1)**i
    ans %= MOD

# 全体NからK個、該当箇所を選ぶ
ans *= nCr(N, K, MOD)
ans %= MOD
print(ans)
