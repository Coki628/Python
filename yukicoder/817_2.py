"""
参考：https://yukicoder.me/problems/no/817/editorial
・はい、ド典型でした。こっちの方針でやれなかったのまじでダメだなぁー。
・二分探索
・K番目をにぶたんで求めるやつね、今回はこれがピタっとハマったね。。
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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

# m円以下の硬貨がK枚以上あるか
def check(m):
    res = 0
    for l, r in AB:
        res += max(min(r, m) - l + 1, 0)
    return res >= K

N, K = MAP()
AB = []
for _ in range(N):
    a, b = MAP()
    AB.append((a, b))

ans = bisearch_min(0, INF, check)
print(ans)
