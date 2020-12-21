# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc025/submissions/5691594
・bitDP、メモ化再帰
・今回は添字の取りうる範囲より遷移する可能性のある場所のが
　全然少ないはずなので、dictでメモを持つ。
・25箇所全部回してたのを、setで必要な所だけ回るようにする。
・pythonはギリギリでTLE、pypyはなぜかWA
"""

import sys
from collections import defaultdict

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

grid=[None]*5
for i in range(5):
    grid[i]=LIST()

used=[-1]*26
unused=set()
for i in range(5):
    for j in range(5):
        if grid[i][j]==0:
            unused.add(i*5+j)
        else:
            used[grid[i][j]]=i*5+j

def check(bit, a):
    h=a//5
    w=a%5
    # 縦位置の確認
    if 1<=h<=3:
        # グリッドでのaから見た上下
        u=(bit>>a-5)&1
        d=(bit>>a+5)&1
        # 上下どちらかだけ埋まってたらNG
        if u^d:
            return 0
    # 横位置の確認
    if 1<=w<=3:
        # グリッドでのaから見た左右
        l=(bit>>a-1)&1
        r=(bit>>a+1)&1
        # 左右どちらかだけ埋まってたらNG
        if l^r:
            return 0
    return 1

memo=defaultdict(lambda: -1)
def rec(bit, cur):
    if memo[bit]!=-1:
        return memo[bit]
    if cur==26:
        return 1
    res=0
    # 置く場所が決まっている時はその遷移だけ確認
    if used[cur]!=-1:
        nxt=used[cur]
        # nxtが既にbitに含まれている時、checkでダメな時はskip
        if not (bit>>nxt)&1 and check(bit, nxt):
            res=rec(bit+(1<<nxt), cur+1)
            res%=MOD
    else:
        # 決まっていない時は25箇所(のうち空いている所だけ)それぞれを試す
        for nxt in unused:
            # nxtが既にbitに含まれている時、checkでダメな時はskip
            if not (bit>>nxt)&1 and check(bit, nxt):
                unused.remove(nxt)
                res+=rec(bit+(1<<nxt), cur+1)
                res%=MOD
                unused.add(nxt)
    memo[bit]=res
    return res

print(rec(0, 1))
