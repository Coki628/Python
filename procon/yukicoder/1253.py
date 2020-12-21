"""
・自力AC！水diff上位の数学問自力突破は嬉しい。
・確率、包除原理
・なるべくシンプルに出せそうなものに分解していく感じ。
・とりあえず反対のが簡単そうなので、全体 - (GCだけ+CPだけ+PGだけ) を考える。
・GCだけ = みんなGかC - みんなG - みんなP とすると、GC^N - G^N - C^N ってなる。
・あとはコードに落とし込むだけ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

def div(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

for _ in range(INT()):
    N, ag, bg, ac, bc, ap, bp = MAP()

    g = div(ag, bg, MOD)
    c = div(ac, bc, MOD)
    p = div(ap, bp, MOD)

    ans = 1 - (pow((g+c), N, MOD) - pow(g, N, MOD) - pow(c, N, MOD)
    + pow((c+p), N, MOD) - pow(c, N, MOD) - pow(p, N, MOD)
    + pow((p+g), N, MOD) - pow(p, N, MOD) - pow(g, N, MOD))
    ans %= MOD
    print(ans)
