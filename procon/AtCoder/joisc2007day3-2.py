# -*- coding: utf-8 -*-

"""
参考：http://algoogle.hadrori.jp/joi/sc2008fraction/
　　　https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A1%E3%83%AC%E3%82%A4%E6%95%B0%E5%88%97
・自力ならず。これは知らなかったらしょうがない。
・ファレイ数列
・a/bとc/dの間に次に出現する分数は(a+c)/(b+d)になる。
　この特性を利用して再帰探索。
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

M, K = MAP()

cnt = 0
ans = ()
def faray(l1, l2, r1, r2):
    global cnt, ans
    m1 = l1 + r1
    m2 = l2 + r2
    if m2 > M:
        return False
    if faray(l1, l2, m1, m2):
        return True
    # K個目に達したら、その時の中間数が答え
    cnt += 1
    if cnt == K:
        ans = (m1, m2)
        return True
    if faray(m1, m2, r1, r2):
        return True
    return False
if faray(0, 1, 1, 1):
    print(*ans)
else:
    print(-1)
