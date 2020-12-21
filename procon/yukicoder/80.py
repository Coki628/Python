"""
・自力AC
・場合分け、と思ったけどしなくてもいいのか。floor(1, 2)したら勝手に0になるもんな。
・合計が決まってて、なるべく積大きくしたい時は、なるべく各値の大きさが同じに近い方がいいってやつ。
　(1*5<2*4<3*3みたいなこと)
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

D = INT()

D //= 2
if D < 2:
    ans = 0
else:
    ans = D // 2 * ceil(D, 2)
print(ans)
