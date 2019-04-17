# -*- coding: utf-8 -*-

"""
参考：http://wk1080id.hatenablog.com/entry/2018/02/05/123131
・AB両方一列に並べちゃっていいんだねー。
"""

import sys
from collections import Counter

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
AB=[(INT(), 'pc') for i in range(N)]+[(INT(), 'plug') for i in range(N)]
AB.sort()

C=Counter()
ans=1
for i in range(N*2):
    # 実は長さは関係なくてPCと電源の並びだけ分かればいい
    _ ,_type=AB[i]
    if C['pc']==C['plug']:
        C[_type]+=1
    # PCが余っている時
    elif C['pc']>C['plug']:
        if _type=='pc':
            C['pc']+=1
        else:
            # 電源が来たらPCを1つ使う
            ans=(ans*C['pc'])%MOD
            C['pc']-=1
    # 電源が余っている時
    elif C['pc']<C['plug']:
        if _type=='plug':
            C['plug']+=1
        else:
            # PCが来たら電源を1つ使う
            ans=(ans*C['plug'])%MOD
            C['plug']-=1
print(ans)
