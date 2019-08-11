# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/ophhdn/items/e62cbf6f57f802f1d8d3
・桁DP系、例えば556なら500,50,6に分けて見ていける。
・でもこの解き方は8進数で余事象取ればいいとか分かってる前提なので、
　自分で導けるか微妙な気がする。
　普通に桁DPで解いてるやつもやってみるかな。。
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

A, B = MAP()

def f(A):
    S = str(A)
    N = len(S)
    res = 0
    # 各桁のループ
    for i, s in enumerate(S):
        num = int(s)
        # 0~その桁の値までのループ
        for j in range(num+1):
            # ループの最後、その桁の値が4か9ならその時点で残りは確定、そうでなければ次の桁へ
            if j == num:
                if j == 4 or j == 9:
                    # ex.) N=3,i=0,j=4 => 468-400+1
                    res += int(S[i:]) - num * 10**(N-i-1) + 1
                    return res
            # 4か9: 全部
            elif j == 4 or j == 9:
                res += 10**(N-i-1)
            # それ以外: 全部 - 8進数で考えた時の全部(使える数字が2つ少ないから)
            else:
                res += 10**(N-i-1) - 8**(N-i-1)
    return res

print(f(B) - f(A-1))
