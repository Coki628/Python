"""
・自力AC！
・区間系
・条件を満たすような区間は求めづらい。逆に条件を満たさない場所に注目すると、
　ある値がない区間に左端・右端が両方収まるものがちょうどそれになるので、そこを数える。
　で、全体から引く。これならうまくデータを隣接リストに持ち替えれば、O(N)で全部分かる。
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

N, M = MAP()
A = [a-1 for a in LIST()]

# 値毎に出現したindexをまとめる
adjli = [[-1] for i in range(M)]
for i, a in enumerate(A):
    adjli[a].append(i)

total = N * (N+1) // 2
ans = [0] * M
for i in range(M):
    adjli[i].append(N)
    cnt = 0
    for j in range(len(adjli[i])-1):
        # 次に同じ値が現れるindexまでの区間長を全部見る
        tmp = adjli[i][j+1] - adjli[i][j] - 1
        cnt += tmp * (tmp+1) // 2
    ans[i] = total - cnt

for a in ans:
    print(a)
