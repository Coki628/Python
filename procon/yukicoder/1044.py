"""
参考：https://www.hamayanhamayan.com/entry/2020/05/02/002420
・自力ならず。数え上げマスターへの道は遠い。。
・数え上げ、nCr系、円環
・円環状なので正直者1を固定して考える。
・正直者と嘘つきが隣接する場所の数がキーになるので、嘘つきグループを挿入する場所の数、でループさせる。
　(ここまでの方針はだいたい出てた。)
・で、各人を区別するので、各隙間に挿入する嘘つきグループが1人以上なのを考えると、写像の3相っぽくて
　包除でループ使うから2重ループだめじゃん、ってなって手詰まりになったんだけど、
　各人の区別は条件に一切影響を与えないので、最後にまとめて掛ければいい、だった。
　そうなるとさっき3相って言ったとこは6相でよくなるので、nCr1発で出る。
・あとはあり得ない条件とかやらないように調整すればOK。
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

N, M, K = MAP()
NM = N + M
mt = ModTools(max(N, M), MOD)

ans = 0
for m in range(1, M+1):
    # 全体 - 正直者と嘘つきの隣接 でKより少ないのはダメ
    if NM - m*2 < K:
        continue
    # m箇所の隙間が選べない
    if m > N:
        break
    # N箇所の隙間からm箇所選ぶ * m箇所に対してM人を最低1人以上入れながら割り振る(写像の6相)
    ans += mt.nCr(N, m) * mt.nCr(M-1, m-1)
    ans %= MOD
# N-1人とM人の並び順を考慮する(条件には関係しないので最後にまとめてでいい)
ans *= mt.fact[N-1] * mt.fact[M]
ans %= MOD
print(ans)
