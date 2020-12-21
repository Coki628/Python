# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-3、自力AC！
・反転(ライツアウト)
・さっき別問題(ABC083d)でWAして、全然方針違ったやつが、こっちにかなり流用できてAC取れた。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# 交互列との一致具合
B = [0] * N
for i in range(N):
    if A[i] == i % 2:
        B[i] = 1
    
# ランレングス圧縮
C = []
cnt = 1
for i in range(N-1):
    if B[i] == B[i+1]:
        cnt += 1
    else:
        C.append(cnt)
        cnt = 1
C.append(cnt)

# 例外処理
if len(C) <= 2:
    print(N)
    exit()

# 各区間iについて切り替えた時に得られる長さから、最大を取る
M = len(C)
ans = 0
for i in range(1, M-1):
    # 両側の区間を繋げられる
    ans = max(ans, C[i]+C[i-1]+C[i+1])
print(ans)
