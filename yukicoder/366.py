"""
・自力AC
・ソート
・幅Kでしか動けないのは、mod K毎に要素を管理すればうまくいく。
・mod K毎の中でソートするのに必要な回数を数える。これは公式解ではバブルソートって
　言ってたけど、ようは転倒数数え上げと同じこと。
　BIT使えばlogで済むけど、制約1000なのでN^2でループして大丈夫。
・最後にmod毎にソートしたやつを元に戻して、それぞれが合ってればOK。ダメなら無理。
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

N, K = MAP()
A = LIST()

li = [[] for i in range(K)]
for i in range(N):
    li[i%K].append(A[i])

cnt = 0
for i in range(K):
    for j in range(len(li[i])):
        for k in range(j+1, len(li[i])):
            if li[i][j] > li[i][k]:
                cnt += 1
    # 実装上はこの後popさせたかったので降順ソートにした。
    li[i].sort(reverse=1)

ans = []
for i in range(N):
    ans.append(li[i%K].pop())
if ans == sorted(A):
    print(cnt)
else:
    print(-1)
