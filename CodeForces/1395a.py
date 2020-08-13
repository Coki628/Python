"""
・回文を作るには、種類が偶数*4か偶数+*3と奇数*1のどっちかにならないといけない。
・で、問題の操作をすると、偶奇は全部反転するだけなので、2回以上やる意味はない。
・最初から0個のボールがあると、1回すら操作できないので、そこは場合分けて処理。
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
    r, g, b, w = MAP()

    for i in range(2):
        C = [0] * 2
        C[r&1] += 1
        C[g&1] += 1
        C[b&1] += 1
        C[w&1] += 1
        if C[1] <= 1:
            Yes()
            break
        if r == 0 or g == 0 or b == 0:
            No()
            break
        r -= 1
        g -= 1
        b -= 1
        w += 3
    else:
        No()
