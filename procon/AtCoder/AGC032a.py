# -*- coding: utf-8 -*-

"""
・400点自力AC！
・AGC参戦でこれは嬉しい。
・逆にBをAに戻していくことを考えたら、後ろから貪欲に操作するだけでうまくいった。
"""

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N=INT()
B=LIST()

ans=deque()
j=N
# 逆にBをAに戻していくことを考える
while len(B):
    # 後ろから見て行って、合うところで操作する
    for i in range(j-1, -1, -1):
        if B[i]==i+1:
            ans.appendleft(B.pop(i))
            j-=1
            break
    else:
        # 操作できるところがなければNG
        print(-1)
        exit()
# Bを空にできたら、成功なので結果出力
for i in range(N):
    print(ans[i])
