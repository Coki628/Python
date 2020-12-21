"""
・ここまで何とか自力AC
・素因数分解、LCM,GCD系
・LCMを持ちたい時素数冪でまとめるといい、ってやつはすぐ出てきて良かった。
・最後の、全ペアのLCMをGCDってのが厄介。結局、全ペア作るってことは、
　2つ以上重複のある素因数なら使えるってことになる。
・なので全体のLCMを取るのと合わせて、kのv乗(k, v)が何回出たかのカウントも集計した。
　後はこの集計に基づいて、最終的なGCDに何が使えるかを決めていく感じ。
・10万のNに√Nかかってるので、計算量は厳しめ。pypyAC2.7秒。(制約3秒)
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def factorize(num: int) -> dict:
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

def lcm(C1, C2):
    for k, v in C2.items():
        C1[k] = max(C1[k], v)
    return C1

N = INT()
A = LIST()

L = Counter()
C = Counter()
for a in A:
    fact = factorize(a)
    L = lcm(L, fact)
    for k, v in fact.items():
        C[(k, v)] += 1
ans = 1
for k, v in L.items():
    cnt = 0
    for i in range(v, 0, -1):
        cnt += C[(k, i)]
        if cnt >= N-1:
            ans *= k**i
            break
print(ans)
