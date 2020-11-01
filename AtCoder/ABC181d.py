"""
・自力AC！
・8の倍数の判定。方法はググる。すると、末尾3桁で8の倍数を作れるかでいいと分かるので、
　3桁以下の8の倍数を列挙して全部見る。予め数えておいたSが持つ数字で何か1つでも作れればOK。
・2桁以下のケースはこれだと判定できないので、そこは場合分けして個別にチェック。(1ペナ)
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

S = input()
N = len(S)
C = Counter(S)

A = []
cur = 8
while cur < 1000:
    A.append(cur)
    cur += 8

ok = False
for a in A:
    a = str(a)
    if len(a) == 1:
        if a == S:
            ok = True
    elif len(a) == 2:
        if a == S or a == S[::-1]:
            ok = True
    else:
        C2 = Counter(a)
        ok = True
        for k, v in C2.items():
            if C[k] < v:
                ok = False
                break
    if ok:
        break
if ok:
    Yes()
else:
    No()
