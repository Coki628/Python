"""
・蟻本演習4-1-3
・包除の練習問題だけど、DPした。大きいサンプル合ったから多分大丈夫だけど、投げてない。
　今の環境だとメモリ64MB制限はpypyMLE確定っぽいしね。。
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
MOD = 10**9 + 7
EPS = 10**-10

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

R, C = MAP()
H, W = MAP()
D, L = MAP()

def gridtoid(i, j):
    return i*W+j

def idtogrid(id):
    return divmod(id, W)

# dp[i][j][k] := iマス目まで見て、j個使った時に、上下左右の状態がkの時の通り数
# ビット対応
# 0: 上, 1: 下, 2: 左, 3: 右
dp = list3d(H*W+1, D+L+1, 1<<4, 0)
dp[0][0][0] = 1
for i in range(H*W):
    h, w = idtogrid(i)
    for j in range(D+L+1):
        for bit in range(1<<4):
            dp[i+1][j][bit] += dp[i][j][bit]
            dp[i+1][j][bit] %= MOD
            if j+1 <= D+L:
                nxbit = bit
                if h == 0:
                    nxbit |= 1<<0
                if h == H-1:
                    nxbit |= 1<<1
                if w == 0:
                    nxbit |= 1<<2
                if w == W-1:
                    nxbit |= 1<<3
                dp[i+1][j+1][nxbit] += dp[i][j][bit]
                dp[i+1][j+1][nxbit] %= MOD
cnt = dp[H*W][D+L][(1<<4)-1]

mt = ModTools(R*C, MOD)
ans = (R-H+1)*(C-W+1) * mt.nCr(D+L, D) * cnt % MOD
print(ans)
