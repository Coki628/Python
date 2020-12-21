# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC、ちょっと時間かかった。
・ライツアウト、ランレングス圧縮、尺取変形
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

N, K = MAP()
S = input()

# ランレングス圧縮
# (後の処理がしやすいように、常に1の区間から始まって1の区間で終わる形にする)
A = []
# 番兵
if S[0] == '0':
    A.append(0)
cnt = 1
cur = S[0]
for i, s in enumerate(S[1:], 1):
    if s != cur:
        A.append(cnt)
        cnt = 1
        cur = s
    else:
        cnt += 1
A.append(cnt)

# 例外処理
if len(A) == 1:
    print(N)
    exit()

# 番兵
if S[-1] == '0':
    A.append(0)

M = len(A)
ans = sm = sum(A[:K*2+1])
for i in range(K*2+1, M):
    sm += A[i]
    # 今回増やすのが0の区間なら、後ろ2つ減る
    if i % 2 == 1:
        if i-K*2 >= 0:
            sm -= A[i-K*2]
        if i-K*2-1 >= 0:
            sm -= A[i-K*2-1]
    ans = max(ans, sm)
print(ans)
