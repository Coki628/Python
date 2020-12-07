"""
参考：https://codeforces.com/blog/entry/85348
・自力ならず。。
・全部マスが埋まってたら、mod 3で斜めに置いてけばいいんだけど。。ってのは考えたのに、
　そこから正解まで辿り着けず。。
・結局、3通り全部試して操作回数がK//3以下になったところでよかった。。
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

def deepcopy(li): return [x[:] for x in li]

for _ in range(INT()):
    N = INT()
    grid = [[]] * N
    K = 0
    for i in range(N):
        grid[i] = list(input())
        K += grid[i].count('X')

    for mod in range(3):
        ans = deepcopy(grid)
        cnt = 0
        for i in range(N):
            for j in range(N):
                if grid[i][j] == 'X' and (i+j) % 3 == mod:
                    ans[i][j] = 'O'
                    cnt += 1
        if cnt <= K//3:
            for i in range(N):
                print(''.join(ans[i]))
            break
