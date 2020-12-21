# -*- coding: utf-8 -*-

"""
参考：https://emtubasa.hateblo.jp/entry/2019/02/12/003250
・問題の意味読み違えてた。
　A*2枚のビスケットがあったら2円分換金していいと思ってた。
　どうりでビスケット増え過ぎると思った。
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

K,A,B=MAP()

# とりあえず換金できるとこまで叩く
rate=B/A
rest=K-(A-1)
bis=A
# 換金やれるだけやって1回残ったら最後叩く
bis+=(B-A)*(rest//2)+(rest%2)

# 上記結果と最後までひたすら叩き続けるやつの大きい方
print(max(bis, K+1))
