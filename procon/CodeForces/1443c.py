"""
・自力AC。ちょっと遅かったな。。
・ソートして累積和と累積max
・最初Bでソートしてしまって合わなかった。
　まあ確かにAの方がmaxだから何個あっても変わらないし、そっち優先のが強い気はする。。
・ちょっとよく考えてみたけど、maxの方を固定すると、それ以下のやつは全部ノーコストで収められるから、
　その条件の中で一番損しない形を作れる、だからmax固定の方を全探索する、って感じかな。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()
    B = LIST()

    AB = list(zip(A, B))
    AB.sort()
    accmx = [0] * (N+1)
    acc = [0] * (N+1)
    for i in range(N):
        accmx[i+1] = max(accmx[i], AB[i][0])
    for i in range(N-1, -1, -1):
        acc[i] = acc[i+1] + AB[i][1]

    ans = INF
    for i in range(N+1):
        ans = min(ans, max(acc[i], accmx[i]))
    print(ans)
