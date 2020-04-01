# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/69464
・自力ならず。。
・こっちの制約弱い方はせめて通したかったね。。
・いい考察ができなくても、こういう全探索はさっと書けるようになりたいねー。
　まあ今回はFに時間割いてしまったからしょうがない部分もあるんだけどさ。
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

N, K = MAP()
A = LIST()

# 取りうる値を全列挙
S = set()
S.add(0)
for a in A:
    while a > 0:
        S.add(a)
        a //= 2

ans = INF
for x in S:
    li = []
    # 各x以上の数について、何回でxにできるかを全て集める
    for a in A:
        cnt = 0
        while a > x:
            a //= 2
            cnt += 1
        if a == x:
            li.append(cnt)
    li.sort()
    if len(li) >= K:
        ans = min(ans, sum(li[:K]))
print(ans)
