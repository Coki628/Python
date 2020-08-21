"""
・制約の大きさから読んだ。
・貪欲シミュレーション
・実は値は0になるばかりで、a2,b1とa1,b2のペアしか増えたり減ったりしない。
・なのでa2,b1をなるべくたくさん作って、b2をなるべく消すためにa0,a2と組ませる。
　残ったb2はa1と組むしかないので、この分を引く。
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
    a0, a1, a2 = MAP()
    b0, b1, b2 = MAP()

    ans = 0
    # a2 * b1
    use = min(a2, b1)
    ans += use * 2
    a2 -= use
    b1 -= use
    # remove b2
    use = min(a0, b2)
    a0 -= use
    b2 -= use
    use = min(a2, b2)
    a2 -= use
    b2 -= use
    ans -= b2 * 2
    print(ans)
