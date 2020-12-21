"""
・なんとか自力AC
・期待値
・最初ダメ元でDPしてみて破滅。M円の時、の状態持ったらそりゃきつい。
・そういえば期待値って1回毎まとめちゃってよかったんじゃないかな、って思って
　毎回その時の期待値みたいのを出してそこを次の遷移元にしたらうまくいった。
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

M, N = MAP()

cur = M
for i in range(N):
    nxt = 0
    nxt += cur*2 / 3
    nxt += (cur+1) / 3
    cur = nxt
print(cur)
