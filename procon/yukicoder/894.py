"""
・自力AC！
・GCD,LCM系、うまく割り算する。
・A,B<=10^18にLCMしたら揺るぎなくでかい数になるなぁと思ったけど、そこはpython多倍長ありがとう。
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

def lcm(x, y): return (x * y) // gcd(x, y)

T, a, b = MAP()

# まず周期の分をまとめてカウントできるように準備する
l = lcm(a, b)
acnt = l // a
bcnt = l // b
# 同時出発分は引く
cnt = acnt + bcnt - 1

d, m = divmod(T, l)
ans = cnt * d
# 端数があれば個別にやる(この間に同時出発は最初の1回しかない)
if m:
    ans += ceil(m, a) + ceil(m, b) - 1
print(ans)
