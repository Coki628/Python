"""
・さくっと自力AC
・再帰探索
・制約が小さいので、区間DPみたいな探索を愚直に全通り見るでOK。
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

S = input()
N = len(S)

se = set()
def rec(l, r, cur):
    if l > r:
        se.add(cur)
        return
    rec(l+1, r, cur+S[l])
    rec(l, r-1, cur+S[r])
    return

rec(0, N-1, '')
ans = len(se)
print(ans)
