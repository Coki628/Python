# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2019/08/25/014348
　　　https://img.atcoder.jp/jsc2019-qual/editorial.pdf
・ある位置を左端として使うか右端として使うか、でしか状態に差が出ない。
　ってことに気付けたらよかったな。。
・そうすると条件を達成できる状態は一意に定まるから、
　あとはそうなる通り数を数えて、最後に順番を考慮してN!やればOK。
"""

import sys
from math import factorial

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

N = INT()
S = input()

# 最初が白だったらどうやっても無理
if S[0] == 'W':
    print(0)
    exit()

# 閉じていない区間の数を保持する
remaining = 1
ans = 1
for i in range(1, N*2):
    # 現在位置が白で残り区間が奇数個 or 現在位置が黒で残り区間が偶数個
    # => この位置は左端にする
    if (S[i] == 'W') ^ (remaining%2 == 0):
        remaining += 1
        # 閉じていない区間数に応じて通り数が増える
        # (今回の左端をどの右端と組ませるかなので、remainingをかける)
        ans *= remaining
        ans %= MOD
    # => この位置は右端にする
    else:
        remaining -= 1
    # 途中で左端が足りなくなったらNG
    if remaining < 0:
        print(0)
        exit()
# 最後に右端が余ったらNG
if remaining != 0:
    print(0)
else:
    # 問題なければ、通り数に順番の分(N!)をかければ答え
    print(ans*factorial(N)%MOD)
