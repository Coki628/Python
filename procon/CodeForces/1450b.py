"""
・自力AC
・ギャグ
・最小回数とか言っといて実は1回じゃないと無理。
・ツイートで、Bはギャグ、みたいなネタバレは食らってしまっていたので、
　そうでなかったら、早く気付けたかどうかは分からないなー。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

for _ in range(INT()):
    N, K = MAP()
    XY = []
    for i in range(N):
        x, y = MAP()
        XY.append((x, y))
    
    cnt = [0] * N
    for i in range(N):
        for j in range(N):
            dist = abs(XY[i][0]-XY[j][0]) + abs(XY[i][1]-XY[j][1])
            if dist <= K:
                cnt[i] += 1
    if max(cnt) == N:
        print(1)
    else:
        print(-1)
