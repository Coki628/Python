"""
・なんとか自力AC。。
・地獄の場合分け
・とりあえず、斜めにいくらでも進めるので、偶奇が合ってればどんなに遠くても2手で、
　合わなくても3手で着く。それより早く着く場合を考える。
・四方に対称で動けるので、一方向だけ考えるといくらか頭のリソースが減る。
・斜めでギリギリまで近づいて、縦横にちょこっと移動するがある。これは1〜2手。
・縦横移動でも、マンハッタン距離3までは1手で行けるのでこれも最初にチェックする。
・これで本番中はAC。でも嘘解法で、after_contestで落ちた。。
・after_contest対策版。ようはマンハッタン距離3の移動2回ってのを考慮できてなかった。。
　これが本番中も入ってたら、通せてなかっただろうなぁ。。
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

r1, c1 = MAP()
r2, c2 = MAP()

ans = INF
if r1 == r2 and c1 == c2:
    ans = min(ans, 0)

if abs(r1-r2) + abs(c1-c2) <= 3:
    ans = min(ans, 1)

if abs(r1-r2) + abs(c1-c2) <= 6:
    ans = min(ans, 2)

distr = abs(r2 - r1)
distc = abs(c2 - c1)
diff = abs(distr - distc)
if diff > 3:
    dist = diff
elif 3 >= diff >= 1:
    dist = 1
else:
    dist = 0
ans = min(ans, 1 + dist)

ans = min(ans, 2 + ((r1+c1)%2 != (r2+c2)%2))
print(ans)
