# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・結構すんなり自力AC
・とはいえ、これ累積max使うのは覚えてたからまあ。
・でも細かい方針をしっかり迷わないで詰められたのは良かったかな。
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

N, C = MAP()
L = [(0, 0)]
for i in range(N):
    x, v = MAP()
    L.append((x, v))
# この後の処理のために右に行った場合も作っておく
R = [(0, 0)]
for i in range(N, 0, -1):
    x, v = L[i]
    R.append((C-x, v))

# 左右について、片側に進んだ場合の累積max
accL = [0] * (N+1)
cur = 0
for i in range(1, N+1):
    x, v = L[i]
    dist = x - L[i-1][0]
    cur +=  v - dist
    accL[i] = max(accL[i-1], cur)
accR = [0] * (N+1)
cur = 0
for i in range(1, N+1):
    x, v = R[i]
    dist = x - R[i-1][0]
    cur +=  v - dist
    accR[i] = max(accR[i-1], cur)

ans = max(max(accL), max(accR))
# 両側に行く場合を確認
for i in range(N+1):
    # 先に左に行って、戻って右へ
    ans = max(ans, accL[i] - L[i][0] + accR[N-i])
    # 先に右に行って、戻って左へ
    ans = max(ans, accR[i] - R[i][0] + accL[N-i])
print(ans)
