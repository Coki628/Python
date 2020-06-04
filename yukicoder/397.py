"""
・自力AC
・ソートの実装
・門松列を無くす、なのでようはソートすればいいんだけど、操作過程を示さないといけない。
・制約は軽いので、ぱっと思いついた実装しやすそうなN^2のソートとかすればいい。
・何かよく分からんけど設定上リアクティブらしいのでflushしないとTLEだった。
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

N = INT()
A = LIST()

ans = []
for i in range(N-1):
    if A[i] > A[i+1]:
        j = i
        while j >= 0 and A[j] > A[j+1]:
            A[j], A[j+1] = A[j+1], A[j]
            ans.append((j, j+1))
            j -= 1
print(len(ans), flush=1)
for a, b in ans:
    print(a, b, flush=1)
input()
