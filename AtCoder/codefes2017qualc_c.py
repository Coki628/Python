# -*- coding: utf-8 -*-

"""
・400点自力AC！
・ほぼ悩まず解けた。想定外がない確証はなかったけど…。
・回文系
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

S=input()

# まずxを除いた状態で回文になっているかどうか
S2=S.replace('x', '')
for i in range(len(S2)//2):
    if S2[i]!=S2[-(1+i)]:
        print(-1)
        exit()

que=deque(S)
cnt=0
while len(que)>1:
    # 前後で一致を確認
    if que[0]!=que[-1]:
        # 合わなければ、xじゃない方にxを足したことにする
        cnt+=1
        if que[0]=='x':
            # 元々xだった方を動かす
            que.popleft()
        elif que[-1]=='x':
            que.pop()
    else:
        # 文字が一致していれば両方動かす
        que.popleft()
        que.pop()
# 最終的に偶数なら空、奇数なら残り1つになったところで終了
print(cnt)
