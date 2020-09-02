"""
・自力AC
・門松列
・こどふぉDiv.2のAとかで出そうなO(1)で条件詰める系。門松列は出ないだろうけど。。
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

for _ in range(INT()):
    a, b, c, costa, costb, costc = MAP()

    ans = INF
    # 大きい方からb,a,c
    cura, curb, curc = a, b, c
    cost = 0
    needa = max(cura-(curb-1), 0)
    cost += needa * costa
    cura -= needa
    needc = max(curc-(cura-1), 0)
    cost += needc * costc
    curc -= needc
    if curc > 0:
        ans = min(ans, cost)
    # 大きい方からb,c,a
    cura, curb, curc = a, b, c
    cost = 0
    needc = max(curc-(curb-1), 0)
    cost += needc * costc
    curc -= needc
    needa = max(cura-(curc-1), 0)
    cost += needa * costa
    cura -= needa
    if cura > 0:
        ans = min(ans, cost)
    # bが最小
    cura, curb, curc = a, b, c
    cost = 0
    if cura == curc:
        if costa <= costc:
            cura -= 1
            cost += costa
        else:
            curc -= 1
            cost += costc
    needb = max(curb-(min(cura, curc)-1), 0)
    cost += needb * costb
    curb -= needb
    if curb > 0:
        ans = min(ans, cost)

    if ans == INF:
        print(-1)
    else:
        print(ans)
