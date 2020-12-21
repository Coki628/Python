# -*- coding: utf-8 -*-

"""
・自力AC
・丁寧に場合分け
"""

import sys
from collections import Counter

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

S1 = input()
S2 = input()
N = len(S1)

C1 = Counter(S1)
C2 = Counter(S2)

def check(C1, C2):
    for k, v in C1.items():
        if C2[k] != v:
            return False
    return True

# まず各文字の種類について個数の一致を確認
if not check(C1, C2):
    NO()
    exit()

# スワップの必要がない箇所を数える
ok = [0] * N
for i in range(N):
    if S1[i] == S2[i]:
        ok[i] = 1
# スワップが必要な回数
cnt = (N - sum(ok)) // 2
if cnt >= 4:
    NO()
elif cnt == 1 or cnt == 3:
    YES()
else:
    # 0,2回スワップしたい場合は、同じ文字で回せる場所があるか調べる
    mx = max(C1.values())
    if mx >= 2:
        YES()
    else:
        NO()
