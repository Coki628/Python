"""
・さくっと自力AC！
・包除原理
・ペンキ→箱、箱→ボールに読み替えると、
　「区別するM個の箱に各1つ以上入れながらN個の区別するボールを振り分ける通り数」
　になるので、これは写像12相の3相そのものとなる。AOJでvarifyしたやつを貼っただけ。
・ド典型だけど、矢印の向きを逆に考えたらすぐに見通しが立ったので、素早く正しい考察ができたのはよかった。
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

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):
        
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

N, K = MAP()

mt = ModTools(max(N, K)+1, MOD)
ans = 0
for i in range(K+1):
    if i % 2 == 0:
        ans += mt.nCr(K, i) * pow(K-i, N, MOD) % MOD
    else:
        ans -= mt.nCr(K, i) * pow(K-i, N, MOD) % MOD
print(ans%MOD)
