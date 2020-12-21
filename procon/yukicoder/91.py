"""
・自力AC
・全種類あるうちは1個ずつでやって、残り2種類になったら後は愚直シミュレーション。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

RGB = LIST()

mn  = min(RGB)
ans = mn
li = []
for i in range(3):
    RGB[i] -= mn
    if RGB[i]:
        li.append(RGB[i])
    
while li:
    if len(li) == 2:
        if li[0] > li[1]:
            li[0], li[1] = li[1], li[0]
        if li[1] >= 3 and li[0] >= 1:
            li[1] -= 3
            li[0] -= 1
            ans += 1
        elif li[1] >= 5:
            li[1] -= 5
            ans += 1
        else:
            break
    else:
        if li[0] >= 5:
            li[0] -= 5
            ans += 1
        else:
            break
print(ans)
