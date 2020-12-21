# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・前から貪欲で確実に最適かちょっと自信なかったけど、無事ACで公式解通り。
"""

import sys

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

def check(A, mod2):
    cur = res = 0
    for i in range(N):
        cur += A[i]
        if i % 2 == mod2:
            # 0以下なら1にする
            if cur <= 0:
                # cur + x = 1 → x = 1 - cur
                A[i] += 1 - cur
                res += 1 - cur
                cur += 1 - cur
        else:
            # 0以上なら-1にする
            if cur >= 0:
                # cur - x = -1 → x = 1 + cur
                A[i] -= 1 + cur
                res += 1 + cur
                cur -= 1 + cur
    return res
    
ans = min(check(A[:], 0), check(A[:], 1))
print(ans)
