# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/titia_til/status/1217127543712600066
・2次元累積和
・Aの最小値をi,Bの最大値をjとする。
・考え方としてはABC106dと同じ感じだと思う。
　不等号の条件に合わせて、条件を満たす側に向かって累積和していく、みたいな。
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

N, M = MAP()

# acc[i][j] := Aの最小値をi,Bの最大値をjとした時の通り数
acc = list2d(N+1, N+1, 0)
# i>jは無理なので、i<=jの範囲で初期化する
for i in range(1, N+1):
    for j in range(i, N+1):
        acc[i][j] = 1

# 初期化と合わせてM回やる
for _ in range(M-1):
    for i in range(1, N+1):
        # i<=jが成り立つならi<=j+1も成り立つので、後ろに足し込んでいく
        for j in range(i, N):
            acc[i][j+1] += acc[i][j]
            acc[i][j+1] %= MOD
    for j in range(1, N+1):
        # i<=jが成り立つならi-1<=jも成り立つので、前に足し込んでいく
        for i in range(j, 1, -1):
            acc[i-1][j] += acc[i][j]
            acc[i-1][j] %= MOD

ans = 0
for row in acc:
    ans += sum(row)
    ans %= MOD
print(ans)
