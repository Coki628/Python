"""
・自力AC。めっちゃ考えて手元で頑張って操作したら気付けた。
　爆速で気付く人は脳みそ何個付いてるんだろう。
・ゲーム系、ギャグ
・ボブは、アリスが最後の1回になるまで何もしないのが最適。このゲームやってて楽しい？
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
    a, b = MAP()

    if a == 1:
        print(0, b)
    else:
        print(a-1, b)
