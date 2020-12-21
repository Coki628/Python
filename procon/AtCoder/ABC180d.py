"""
・自力AC！
・こどふぉにありがちそうな貪欲詰め。
・掛ける方は最低でも2で値がすぐ大きくなるから全パターン試せる。
　その掛けた回数の時の残りの足せる回数は割り算ゴニョゴニョすればO(1)なので、やる。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

x, y, a, b = MAP()

ans = 0
st = x
exp = 0
ans = (y-st-1) // b
while st < y:
    st *= a
    exp += 1
    ans = max(ans, (y-st-1) // b + exp)
print(ans)
