"""
・さくっと自力AC
・等差数列、二分探索
・一番でかい丸太で貪欲に小さいやつを作るがよさそう、ってなる。
・どこまで作れるかは初項1公差1の等差数列の和がN+1以内で何項まで作れるかでいいので、
　頭よく計算もありそうだったけど、これもまた迷わずにぶたん貼って殴る。
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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N = INT()

def check(m):
    return (m+1)*m // 2 <= (N+1)

res = bisearch_max(0, INF, check)
ans = N - res + 1
print(ans)
