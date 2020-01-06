# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・累積和応用
・当時は出なかった考察みたいなので、ちゃんと出来てまあ嬉しい。
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

N = INT()
A = LIST()

# +-どっちに揃えるかを決める
mn = min(A)
mx = max(A)
if abs(mx) >= abs(mn):
    flag = 1
    base = mx
    baseidx = A.index(mx)
else:
    flag = 0
    base = mn
    baseidx = A.index(mn)

# 符号を揃える
ans = []
for i in range(N):
    if i == baseidx:
        continue
    A[i] += base
    ans.append((baseidx, i))

# 正負によって左右から累積和っぽくする
if flag:
    for i in range(N-1):
        A[i+1] += A[i]
        ans.append((i, i+1))
else:
    for i in range(N-2, -1, -1):
        A[i] += A[i+1]
        ans.append((i+1, i))

print(len(ans))
for a, b in ans:
    print(a+1, b+1)
