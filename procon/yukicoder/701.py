"""
・自力AC
・実装
・ランダム生成のやつ、問題解くのに使ったのは初めてかも。
"""

import sys
from random import randrange

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

M = 26
L = 20
NG = 13
# 始めと終わりの文字毎で文字列を管理する
table = list3d(M, M, 20, 0)
for i in range(M):
    for j in range(M):
        table[i][j][0] = i
        table[i][j][-1] = j

cur = 0
ans = []
for i in range(N-1):
    # n以外から適当に選ぶ
    while 1:
        nxt = randrange(0, M)
        if nxt != NG:
            break
    for i in range(1, L-1):
        # z未満なら追加して1つ進める
        if table[cur][nxt][i]+1 < M:
            ans.append(''.join(map(chr, [num+97 for num in table[cur][nxt]])))
            table[cur][nxt][i] += 1
            break
    cur = nxt
# 最後はnにする
nxt = NG
ans.append(''.join(map(chr, [num+97 for num in table[cur][nxt]])))

for s in ans:
    print(s)
