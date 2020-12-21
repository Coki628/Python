"""
・自力AC
・最初、部分和DPで攻めてしまって、めっちゃ数がでかくなりサンプル後半が合わず、
　順番違いを数えないことに気付き正気に戻った。3乗間に合うので雑に全部数えればOK。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

ans = 0
for i in range(N+1):
    for j in range(i+1):
        for k in range(N+1):
            if i*5 + j*2 + k*3 == N:
                ans += 1
print(ans)
