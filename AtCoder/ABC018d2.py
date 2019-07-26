# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc018
・半分全列挙
・女子だけ決めれば、男子を貪欲に選べる。
・pythonはTLE、pypyでAC0.7秒。
"""

import sys
from itertools import combinations

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

N,M,P,Q,R=MAP()
XYZ=[]
for i in range(R):
    x,y,z=MAP()
    XYZ.append((x-1, y-1, z))

# 女子だけ全組み合わせをやる(半分全列挙)
ans=0
for girls in combinations(range(N), P):
    girls=set(girls)
    # その男子を入れた場合に得られる幸福度を格納する
    boys=[0]*M
    for i in range(R):
        x,y,z=XYZ[i]
        # 渡す女子がいる場合のみ考える
        if x in girls:
            boys[y]+=z
    # 多い方からQ人の合計を取る
    boys.sort(reverse=True)
    ans=max(ans, sum(boys[:Q]))
print(ans)
