# -*- coding: utf-8 -*-

"""
・終了後自力AC。時間なかった訳ではないので(45分くらいはあった)、通したかったなー。
・こどふぉあるあるの条件に合う実装頑張って作る系
・色々試行錯誤したけど、まとめると、
　・末尾桁を探す
　・2番目桁を探す
　・先頭0に対応する
　を順番にやっていけばOK。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = list(map(int, input()))
N = len(S)

# 1桁だと25の倍数そもそも無理なので先に弾く
if N <= 1:
    print(-1)
    exit()

# 末尾から2番目桁をnum1、末尾をnum2として、実現させる最小操作回数
def calc(num1, num2):
    # 最初からOK
    if S[-2] == num1 and S[-1] == num2:
        return 0
    # 1つスワップさせるだけ
    if S[-1] == num1 and S[-2] == num2:
        return 1
    # 末尾桁について、該当の数字を探しに行って一番近い所から取ってくる
    res = 0
    T = S[:]
    if T[-1] != num2:
        for i in range(N-2, -1, -1):
            if T[i] == num2:
                # 見つけたら該当位置までスワップさせてくる
                for j in range(i, N-1):
                    T[j], T[j+1] = T[j+1], T[j]
                    res += 1
                break
        else:
            return INF
    # 2番目桁についても同様に探してくる
    if T[-2] != num1:
        for i in range(N-3, -1, -1):
            if T[i] == num1:
                for j in range(i, N-2):
                    T[j], T[j+1] = T[j+1], T[j]
                    res += 1
                break
        else:
            return INF
    # 上記2つの操作で先頭桁が0になってしまった場合
    if T[0] == 0:
        for i in range(N-2):
            # 一番近い0ではない桁から取ってくる
            if T[i] != 0:
                break
            else:
                res += 1
        else:
            return INF
    return res

# 25の倍数は末尾2桁が50, 25, 75, 00のどれかなので全部試す
ans = min(calc(5, 0), calc(2, 5), calc(7, 5), calc(0, 0))
if ans == INF:
    print(-1)
else:
    print(ans)
