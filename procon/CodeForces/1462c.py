"""
・自力AC
・順列全探索
・結局のところ0〜9の10種類の数字で重複なくしか置けないなら、10!(と9!以下も全部)くらいしかないから、
　何とか全探索できるかなぁってやった。pypyAC1.06秒。制約1秒ならダメだったね。。
・想定解は貪欲っぽい。TLに流れてたの見たら、9から順に足りない限り使っていけばいいって感じで、
　確かにそれだとビット上から順に使ってく時みたいにうまくいくっぽいね。。
"""

import sys
from itertools import permutations

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

D = {}
for d in range(1, 11):
    for perm in permutations(range(10), d):
        sm = sum(perm)
        if sm not in D:
            D[sm] = ''.join(map(str, perm))

for _ in range(INT()):
    N = INT()

    if N in D:
        print(D[N])
    else:
        print(-1)
