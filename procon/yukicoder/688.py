"""
・自力AC
・場合の数、全探索、構築
・数列の長さは30以内と小さいので、0,1それぞれ何個あるかを全探索でOK。
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

def nC2(n):
    return n*(n-1) // 2

K = INT()
N = 30
for i in range(1, N+1):
    for j in range(i+1):
        k = i - j
        # 0がj個、1がk個ある場合の通り数を調べる
        cnt = 2**j * nC2(k)
        if cnt == K:
            ans = [0] * j + [1] * k
            print(i)
            print(*ans)
            exit()
