"""
・自力AC
・ランレングス圧縮
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def RLE(data):
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

for _ in range(INT()):
    a, b = MAP()
    S = input()

    rle = RLE(S)
    # 区間1つは場合分け
    if len(rle) == 1:
        if rle[0][0] == '0':
            print(0)
        else:
            print(a)
    else:
        # 両端の0は必要ないので省く
        if rle[-1][0] =='0':
            rle.pop()
        if rle[0][0] == '0':
            rle.pop(0)
        ans = 0
        for s, cnt in rle:
            # 0区間は、そのままか1区間と繋げるか、いい方にする
            if s == '0':
                ans += min(0, cnt*b-a)
            else:
                ans += a
        print(ans)
