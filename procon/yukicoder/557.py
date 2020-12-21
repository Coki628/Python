"""
・結構さくっと自力AC！
・数え上げ
・1桁は1,8の2通りだけとして例外処理しておく。それ以外は、前後でペアになる位置毎に考えて、
　基本的には1-1,8-8,0-0,6-9,9-6の5通り、先頭桁だけは0-0を除いた4通りとする。
・これに、桁数が奇数の時は真ん中として1,8,0を使う3通りも加味する。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

if N == 1:
    print(2)
    exit()

ans = pow(5, N//2-1, MOD) * 4 % MOD
if N % 2 == 1:
    ans *= 3
    ans %= MOD
print(ans)
