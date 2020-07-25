"""
・自力AC
・幾何、円
・順番に横並びに置いていって、x座標が1500をはみ出すところでy座標を動かして次の段って感じ。
　結構スカスカで適当な置き方だけど、これでも十分収まった。
・一応考えたこととしては、最大の円k=100でも直径で200*200なので、2000*2000の領域があれば100個置ける。
　実際は2*2,4*4,6*6と小さいやつがたくさんあるので、まあ1500*1500あれば雑でも全部置けるだろ、みたいな感じ。
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

ans = []
k = 1
# 円を置く左下座標を持っておく
basey = basex = 0
while k <= 100:
    # x座標が1500をはみ出すところでy座標を動かして次の段
    if basex + k*2 > 1500:
        # 今置いてある一番大きい円と干渉しない位置までyを動かす
        basey += (k-1) * 2
        basex = 0
    x = basex + k
    y = basey + k
    ans.append((x, y))
    # 基本的にはyは固定でxだけ動かしていく
    basex += k * 2
    k += 1
for xy in ans:
    print(*xy)
