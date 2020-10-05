"""
・自力AC！
・回文行列、中央値
・回文にしなければいけないので、対角にある4つは基本的に同じ値にしないといけない。
　なのでその4つを取ってきて、一番動かす和が少ない値にする、をやる。
・と思ったら合わない、、なんでだ。。しばらく悩む。
・で、結局適切な値を見つけるのに、中央値じゃなくて平均値を使ってたからダメだったんだね。気付いてよかった。
・あと一応、奇数長のところは片側についてだけチェックするから場合分けね。
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

for _ in range(INT()):
    N, M = MAP()

    grid = [[] for i in range(N)]
    for i in range(N):
        grid[i] = LIST()

    ans = 0
    for i in range(N//2):
        k = N - i - 1
        for j in range(M//2):
            l = M - j - 1
            li = [grid[i][j], grid[i][l], grid[k][j], grid[k][l]]
            li.sort()
            cent = li[1]
            ans += abs(cent-grid[i][j]) + abs(cent-grid[i][l]) + abs(cent-grid[k][j]) + abs(cent-grid[k][l])
 
    if N % 2 == 1:
        i = N // 2
        for j in range(M//2):
            l = M - j - 1
            ans += abs(grid[i][j]-grid[i][l])
    if M % 2 == 1:
        for i in range(N//2):
            k = N - i - 1
            j = M // 2
            ans += abs(grid[i][j]-grid[k][j])

    print(ans)
