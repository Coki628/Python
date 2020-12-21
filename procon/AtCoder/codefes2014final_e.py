# -*- coding: utf-8 -*-

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

N = INT()
A = LIST()

# そもそも足りない
if N <= 2:
    print(0)
    exit()

# 同値ではなくなる所まで進める
i = 0
while i < N-1 and A[i] == A[i+1]:
    i += 1
if i == N-1:
    print(0)
    exit()

# 上がってるか下がってるか
if A[i] < A[i+1]:
    is_up = 1
else:
    is_up = 0
cnt = 2
for j in range(i+1, N-1):
    if A[j] == A[j+1]:
        continue
    # 向きが変わる所で、フラグ切り替えとカウント
    elif (A[j] < A[j+1]) ^ is_up:
        is_up = 1 - is_up
        cnt += 1
# 3点以上になっていればOK
if cnt >= 3:
    print(cnt)
else:
    print(0)
