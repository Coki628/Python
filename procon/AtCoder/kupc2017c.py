# -*- coding: utf-8 -*-

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

A=INT()
S=input()
N=len(S)

# まずハッシュ値を出す
hashed=0
for i in range(N):
    hashed+=A**(i+1)*(ord(S[i])-96)

ans=''
for i in range(N):
    # 次が26以下になる時はそれで最後
    if hashed//A**(i+1)<=26:
        tmp=hashed//A**(i+1)
        ans+=chr(tmp+96)
        break
    # 大きい文字から試していく
    for j in range(26, 0, -1):
        tmp=hashed-A**(i+1)*j
        # 次の文字の時に割り切れるかどうか
        if tmp%A**(i+2)==0:
            # 大丈夫ならその文字で確定
            ans+=chr(j+96)
            hashed=tmp
            break
print(ans)
