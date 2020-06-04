"""
・さくっと自力AC
・N進数系
・N進・10進変換もだいぶ慣れてきたかな。
"""

import sys
from string import digits

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

N = INT()
A = [list(input()) for i in range(N)]

for i in range(N):
    for j in range(len(A[i])):
        if A[i][j] in digits:
            A[i][j] = int(A[i][j])
        else:
            A[i][j] = ord(A[i][j]) - 55

ans = INF
for S in A:
    # 使われている最大+1が解釈できる最小
    n = max(S) + 1
    num = 0
    # 各桁をn進数として10進数に変換
    for k, s in enumerate(S[::-1]):
        num += s*n**k
    ans = min(ans, num)
print(ans)
