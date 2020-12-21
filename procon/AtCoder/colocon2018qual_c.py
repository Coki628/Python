# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/colopl2018-qual/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2017/12/10/031120
・全探索の応用
・奇数はビット全探索、偶数は1つ使うor使えないの二択しかないので普通のループでいける。
・互いに素
・今回は対ごとに素(どの二つも互いに素)である必要があるので、gcd(a,b,c)ではダメで
　個別に2つずつ見ていく必要がある。
・O(2^((B-A)/2)*(((B-A)/2)^2)/2=2^18*18^2/2=4200万くらいはpypyでもTLE
　(それに加えてgcd()もO(1)じゃなさそうだし)
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from functools import reduce

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

A,B=MAP()

odd=[]
even=[]
for i in range(A, B+1):
    if i%2==0:
        even.append(i)
    else:
        odd.append(i)
# とりあえず偶数全部1つだけ選ぶ分と0の分
ans=len(even)+1

valid=[]
# 奇数はビット全探索
for i in range(1<<len(odd)):
    check=[]
    for j in range(len(odd)):
        if i>>j&1:
            check.append(odd[j])
    if len(check)==0: continue
    ok=True
    # この二重ループで各2値を全てgcdしていく
    for j in range(len(check)):
        for k in range(j+1, len(check)):
            if gcd(check[j], check[k])!=1:
                ok=False
    if ok:
        valid.append(check)
        ans+=1

for num in even:
    for check in valid:
        ok=True
        for i in range(len(check)):
            # ある偶数と、奇数でうまくいった組み合わせ内の各値を突き合わせていく
            if gcd(num, check[i])!=1:
                ok=False
        if ok:
            ans+=1
print(ans)
