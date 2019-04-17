# -*- coding: utf-8 -*-

"""
・Nim、XOR系
・ちょっと集計のとこ整理
"""

import sys
from functools import reduce
from operator import xor

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
A=[INT() for i in range(N)]

mnbit=set()
for i, a in enumerate(A):
    bina=format(a, 'b')
    # 2進で最初に1になる桁を、各値について見る
    mnbit.add(len(bina)-bina.rfind('1'))

# 現状のA全部XOR
res=format(reduce(xor, A, 0), 'b')
cnt=0
ln=len(res)
for i in range(ln):
    idx=ln-i
    # 大きい桁から見て、1になっているところについて処理する
    if res[i]=='1':
        # その桁用の値があるか
        if idx in mnbit:
            # あればそれを1減らせば、その桁より下が全部反転する
            res=format(int(res, 2)^int('1'*idx, 2), '0'+str(ln)+'b')
            cnt+=1
        else:
            # 必要な桁用の値がなければNG
            print(-1)
            exit()
print(cnt)
