"""
参考：https://yukicoder.me/problems/no/751/editorial
　　　https://www.geisya.or.jp/~mwm48961/kou3/bunsuushiki3.htm
・自力ならず。。
・繁分数
・って言葉が出てこなくて、なかなかうまくググれなかった。
・横棒が短い所から計算する。下が短い方が分かりにくい。
　最終的に、a/b/c/d/e/f → bdf/ace みたいになる。
・上が短い方は前から割り算なので、最初だけ分子であと全部分母。
・約分はGCDでサクッと整える。マイナスでバグると嫌なのでそこはabsしとく。
・マイナスは分子固定とのことなので、そこは場合分けして後付け。
・はぁ、数学ゲーおつ。。。
"""

import sys
from math import gcd

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
M = INT()
B = LIST()

x = y = 1
x *= A[0]
for i in range(1, N):
    y *= A[i]

for i in range(M):
    if i % 2 == 1:
        x *= B[i]
    else:
        y *= B[i]

if (x < 0) ^ (y < 0):
    x = -abs(x)
    y = abs(y)
else:
    x = abs(x)
    y = abs(y)
g = gcd(abs(x), abs(y))
x //= g
y //= g
print(x, y)
