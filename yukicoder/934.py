"""
・自力AC！
・インタラクティブ問題
・少ない方からでなく、多い方から1つずつ減らしていく方針で行けば必要なものを特定できる。
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

A = list(range(1, N+1))
i = 0
while 2 < len(A) and i < len(A):
    tmp = A.pop(i)
    print('? {0}'.format(len(A)), flush=1)
    print(*A, flush=1)
    res = INT()
    if not res:
        A.insert(i, tmp)
        i += 1
print('! {0}'.format(len(A)), flush=1)
print(*A, flush=1)
