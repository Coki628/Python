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

year = INT()

# 後の条件を優先なので、elifにせずに各条件節でフラグを切り替える
is_uru = False
if year%4 == 0:
    is_uru = True
if year%100 == 0:
    is_uru = False
if year%400 == 0:
    is_uru = True

if is_uru:
    YES()
else:
    NO()
