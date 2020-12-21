# -*- coding: utf-8 -*-

"""
・なんと初の700点自力AC！
・これは嬉しいー。けど、そこまで時間かからず法則読めたから、これなら本番ちゃんと見れば良かった…。
"""

import sys

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

ans=[]
# 偶奇で分ける
if N%2==0:
    # ペアがN+1にならない場所全部に辺を張る
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if i+j!=N+1:
                ans.append((i, j))
else:
    # 一番大きいNには全部から辺を張る
    for i in range(1, N):
        ans.append((i, N))
    # 残りはペアがNにならない場所全部に辺を張る
    for i in range(1, N):
        for j in range(i+1, N):
            if i+j!=N:
                ans.append((i, j))

print(len(ans))
for edge in ans:
    print(*edge)
