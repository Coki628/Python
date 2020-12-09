"""
・自力AC！
・三角形グリッド
・どういう状態か考えると正三角形が敷き詰められた平面上を動き回る感じになるけど、
　これを、位置(移動回数)の偶奇で、左右下か左右上にしか動けないグリッド上と捉えると、うまく処理できた。
・辺の位置によって次に動きたい方向が分かるので、
　移動した時に辺a,b,cがそれぞれどの位置になっていくのかを考える。
・動く方向が上下の時はそれぞれの位置は変わらないので何もしなくてOK。
　問題は左右に動く時だけど、これは左に動く時は、左を右に、右を上下に、上下を左に、とやって、
　右に動く時には、右を左に、左を上下に、上下を右に、とやるといい。
　実装上はこれを逆の順序で当てはめていけば依存性に問題なく処理できる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

S = input()
N = len(S)

h = w = 0
visited = set()
visited.add((h, w))
pos = {'a': 'ud', 'b': 'l', 'c': 'r'}
for i in range(N):
    nxd = pos[S[i]]
    if nxd == 'ud':
        if i % 2 == 0:
            h -= 1
        else:
            h += 1
    elif nxd == 'l':
        # ud -> l, r -> ud, l -> r
        for c in 'abc':
            if pos[c] == 'ud':
                pos[c] = 'l'
        for c in 'abc':
            if pos[c] == 'r':
                pos[c] = 'ud'
        pos[S[i]] = 'r'
        w -= 1
    else:
        # ud -> r, l -> ud, r -> l
        for c in 'abc':
            if pos[c] == 'ud':
                pos[c] = 'r'
        for c in 'abc':
            if pos[c] == 'l':
                pos[c] = 'ud'
        pos[S[i]] = 'l'
        w += 1
    visited.add((h, w))

ans = len(visited)
print(ans)
