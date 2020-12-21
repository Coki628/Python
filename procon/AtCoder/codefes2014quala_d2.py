# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/codefestival2014quala
・公式解も一応見ておく。
・条件を決めて全探索
・こういう、条件をしっかり詰めて貪欲みたいの、難しいよなー。
・聞けば確かにそれはそう、なんだけどね。。
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

A, K = MAP()
total = A
A = str(A)
N = len(A)

ans = INF
for p in range(N+1):
    for q in range(10):
        for r in range(10):
            tmp = []
            # p桁目まではAと合わせる
            for i in range(p):
                tmp.append(A[i])
            # p+1桁目をqとする
            tmp.append(str(q))
            # p+2桁目以降をrとする
            for i in range(p+1, N):
                tmp.append(str(r))
            # 使った数字をカウント
            used = set()
            for a in tmp:
                # 最上位桁の0は使用済にしない
                if a == '0' and len(used) == 0:
                    continue
                used.add(a)
            # 使った数字がK個以内なら答えの候補になる
            if len(used) <= K:
                tmp = int(''.join(tmp))
                ans = min(ans, abs(total-tmp))
print(ans)
