"""
・自力AC
・ビットの考え方、組み合わせ
・しばらく観察すると、関数fはどんどん半分にしていった時に0になるまでの奇数になる回数、って感じだと分かった。
・これは2冪で考えると、ビットの立っている数になる。
・ここまで分かると後はnCrの標準的な組み合わせ。
・総和の方は、各ビット出現回数は均等だと思ったので、通り数に立ってるビット数掛けて全ビットで割った。
　これが各ビットが使われた回数になる。
・公式解は「あるビットが選ばれた時のその他のビットが取りうる通り数」から出現回数を割り出していた。(頭いい)
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

def nCr(n, r):
    from math import factorial

    if n < r: return 0
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

def f(n):
    if n == 0:
        return 0
    else:
        return f(n//2) + (n&1)

x = INT()

if x > 31:
    print(0, 0)
    exit()

cnt = nCr(31, x)
each = cnt * x // 31
sm = 0
for k in range(31):
    sm += 2**k * each
print(cnt, sm)
