"""
・自力AC
・なんかめんどくさい連立方程式とか立てるのかと思ったら、
　よく考察したら場合分けでいけた。
・みんな自分以外は正確にカウントしているので、結局Aの各値は高々2種類。
　多い方が鶴(カウントしてない足が2本だけ)、逆が亀(カウントしてない足が4本)。
　全部同値だと、どっちだかは分からないけど、全員鶴だと仮定して計算が合うか、
　みたいな条件処理つければどっちだか分かる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

mn = min(A)
mx = max(A)
ans = [0] * 2
if mn == mx:
    if (mn + 2) // N == 2:
        ans[0] = N
    else:
        ans[1] = N
else:
    for a in A:
        if a == mx:
            ans[0] += 1
        else:
            ans[1] += 1
print(*ans)
