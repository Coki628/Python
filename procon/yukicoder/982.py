"""
・なんとか自力AC
・整数系、部分和DP(簡易版)
・数学考察強いと、O(AB)じゃなくてO(B)とかO(1)？でも解けるみたい。。
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

a, b = MAP()

# a,bが互いに素でなければNG
if gcd(a, b) != 1:
    print(-1)
    exit()

# LCMまでの数で踏めない数を数える
N = lcm(a, b)
dp = [0] * (N+1)
dp[0] = 1
for i in range(N+1):
    if dp[i]:
        if i+a <= N:
            dp[i+a] = 1
        if i+b <= N:
            dp[i+b] = 1
ans = dp.count(0)
print(ans)
