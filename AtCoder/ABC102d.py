# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc100/editorial.pdf
　　　https://www.youtube.com/watch?v=UX4AuiCVtN4
　　　https://atcoder.jp/contests/arc100/submissions/2857937
・累積和、二分探索
・区間4つあっても、真ん中を決めれば前と後ろの境界は一意に定まる。
・公式解は尺取言ってたけど、二分探索にした。
　多分、こっちのが実装分かりやすいと思う。
"""

import sys
from itertools import accumulate
from bisect import bisect_left

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

N=INT()
A=LIST()

cmsm=[0]+list(accumulate(A))
ans=INF
for i in range(2, N-1):
    # 左
    S1=cmsm[i]
    idx1=bisect_left(cmsm, S1/2)
    # 境界の手前と比べて、差が小さい方を取る
    if abs(cmsm[idx1-1]-S1/2)<abs(cmsm[idx1]-S1/2):
        idx1-=1

    # 右
    S2=cmsm[N]-cmsm[i]
    idx2=bisect_left(cmsm, S1+S2/2)
    if abs(cmsm[idx2-1]-(S1+S2/2))<abs(cmsm[idx2]-(S1+S2/2)):
        idx2-=1

    p=cmsm[idx1]-cmsm[0]
    q=cmsm[i]-cmsm[idx1]
    r=cmsm[idx2]-cmsm[i]
    s=cmsm[N]-cmsm[idx2]
    mx=max(p, q, r, s)
    mn=min(p, q, r, s)
    ans=min(ans, mx-mn)
print(ans)
