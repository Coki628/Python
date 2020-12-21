# -*- coding: utf-8 -*-

"""
・400点自力AC
・桁和系
・場合分けしたけど考慮足りてないかなーと思ったら通った。
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

digism=0
for s in str(N):
    digism+=int(s)

ans=0
# 桁和が10未満かで場合分け
if digism<10:
    ans=digism
    # Nと最小が被ったら1つ大きいやつにする
    if ans==N:
        ans='1'+str(N-1)
else:
    num9=digism//9
    rest=digism%9
    ans=int(str(rest)+('9'*num9))
    # Nと最小が被ったら1つ大きいやつにする
    if ans==N:
        # 999...かどうかで場合分け
        if rest!=0:
            ans=str(rest+1)+'8'+('9'*(num9-1))
        else:
            ans='18'+('9'*(num9-1))
print(ans)
