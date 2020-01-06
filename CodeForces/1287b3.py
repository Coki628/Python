# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1287/submission/68255688
・似たような方針なのに速度が倍近く違ったので気になって検証。
・結果的には意外な所が原因だった。
・dictの処理をordとchrでやるようにしたら、2.8→1.1秒くらいまで変わった。
・そんなにdictのアクセスが遅い訳ないんだけど、うーん何でだろう。。
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

N, K = MAP()
A = [input() for i in range(N)]

S = set(A)
ordsm = ord('S') + ord('E') + ord('T')

cnt = 0
for i in range(N):
    a1 = A[i]
    for j in range(i+1, N):
        a2 = A[j]
        a3 = ''
        for k in range(K):
            if a1[k] == a2[k]:
                a3 += a1[k]
            else:
                key = ordsm - ord(a1[k]) - ord(a2[k])
                a3 += chr(key)
        if a3 in S:
            cnt += 1
print(cnt // 3)
