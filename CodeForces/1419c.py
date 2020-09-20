"""
・自力AC！
・結局のところ0,1,2のどれか。こどふぉっぽい。
・何も変えなくてよければ0、全部変えてぴったり合うか、元々変えなくていいのが1つでもあれば1、
　それ以外なら2回。
・最初WAで結構悩んで、元々変えなくていいのが1つでもあれば、が抜けてた。よく気付いた。
・基本方針として、とりあえず最後の1個以外全部はまとめて変えられて、合わなければ最後の1個だけ個別に、
　だったんだけど、1つでも元々Xのやつがあれば、そいつを最後の1個として使えばいいんだよね。
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
    N, X = MAP()
    A = LIST()

    se = set(A)
    if len(se) == 1 and list(se)[0] == X:
        print(0)
        continue
    cnt = 0
    for a in A:
        cnt += X - a
    if cnt == 0 or X in se:
        print(1)
    else:
        print(2)
