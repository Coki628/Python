"""
参考：https://drken1215.hatenablog.com/entry/2019/04/20/170200
・蟻本演習4-1-3
・nCr数え上げ、スターリング数
・自力無理だったー。解説見てもしばらく考えてやっと理解って感じ。。
・kグループに分けるを全探索は考えたんだけど、夫婦一緒にするn組と別々にするN-n組のnを全探索、
　が出なかったー。言われれば、確かにありそうなやり方だな。。
　これなら事情の違う一緒にする方と別々にする方で独立に計算できてうまくいくね。
・n組についてはスターリング数そのものなので、前計算でDPしておく。
・N-n組については、別々にしたいのでk^(N-n)と(k-1)^(N-n)にするんだけど、
　この時区別しない箱だから重複度k!で割らないといけない気がしてしまったんだけど、
　こっちの組を計算してる時にはもう区別できるkグループがn組の分け方によって出来上がってるから、
　区別しない箱とか考えなくていいんだね。。。
・数え上げむずい。ちょっと難易度上がると、すぐ考察進まなくなっちゃうなぁ。。
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

    def nHr(self, n, r):
        """ 重複組み合わせ """

        return self.nCr(r+n-1, r)

    def nPr(self, n, r):
        """ 順列 """

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD

# N個の区別する玉を、K個の区別しない箱に入れる(各箱に玉1個以上)
def stirling(N, K):
    dp = list2d(N+1, K+1, 0)
    dp[0][0] = 1
    for i in range(1, N+1):
        for k in range(1, K+1):
            dp[i][k] = dp[i-1][k-1] + k*dp[i-1][k]
            dp[i][k] %= MOD
    return dp

N = INT()

mt = ModTools(N, MOD)
S = stirling(N, N)

ans = 0
# k個のグループに分ける
for k in range(1, N+1):
    # n組は同じグループ、m組は別グループに入れる
    for n in range(N+1):
        m = N - n
        # N組から、同じグループにするn組を選ぶ
        # * n組を各グループ1組以上いれながらkグループに分ける(スターリング数)
        # * m組は違うグループに入れる(片方はkから自由に選び、相方はそれ以外のk-1から選ぶ)
        ans += mt.nCr(N, n) * S[n][k] * pow(k, m, MOD)*pow(k-1, m, MOD)
        ans %= MOD
print(ans)
