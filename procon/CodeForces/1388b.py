"""
・変換の条件から、9→1001,8→1000,7→111,,,となるけど、最大にしたいので桁的に8以上は必須で、
　最終的に削られる位置なら9でなく8にしていい。
・削られるかどうかは桁数Nを割る4切り上げすると分かる。そこまでは8にしてあとは9に。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()

    cnt8 = ceil(N, 4)
    cnt9 = N - cnt8
    ans = '9'*cnt9 + '8'*cnt8
    print(ans)
