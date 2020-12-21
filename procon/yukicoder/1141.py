"""
・自力AC！
・2次元累積積、0のコーナーケース
・問題見て、2次元累積和の積になったやつかなー、実装だるめだけどまあ解けるでしょ、
　って取り掛かったら意外と手こずる。。
・WAが出るのは0のマスを正しく考慮してないから。原因は分かったけどちょっと方針が立たなくて、
　「累積積 ゼロ」ってググったら、胡散臭い翻訳サイトに一応手がかりになる方針が載ってた。
・0が出たところは、累積積とは区別して場所を取っておく。
　で、これも累積和して区間での出現数を取れるようにしておく。
・これでその時使う区間積にゼロが含まれるかどうかもチェックできるので、クエリに正しく答えられる。
"""

import sys

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

def div(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

H, W = MAP()
grid = list2d(H+2 ,W+2, 1)
zeros = list2d(H+2, W+2, 0)
for i in range(1, H+1):
    grid[i] = [1] + LIST() + [1]
    for j in range(1, W+1):
        if grid[i][j] == 0:
            # ゼロがあったら、別途場所は覚えておいて、累積積の方は1にしておく
            zeros[i][j] = 1
            grid[i][j] = 1

for i in range(H+2):
    for j in range(W+1):
        grid[i][j+1] *= grid[i][j]
        grid[i][j+1] %= MOD
        zeros[i][j+1] += zeros[i][j]
for j in range(W+2):
    for i in range(H+1):
        grid[i+1][j] *= grid[i][j]
        grid[i+1][j] %= MOD
        zeros[i+1][j] += zeros[i][j]

Q = INT()
for _ in range(Q):
    r, c = MAP()

    # 塗りつぶされない4領域の区間積
    rd = div(grid[H+1][W+1] * grid[r][c], grid[H+1][c] * grid[r][W+1], MOD)
    ru = div(grid[r-1][W+1], grid[r-1][c], MOD)
    ld = div(grid[H+1][c-1], grid[r][c-1], MOD)
    lu = grid[r-1][c-1]

    # ゼロの出現位置を確認する区間和
    rd_zero = zeros[H+1][W+1] + zeros[r][c] - zeros[H+1][c] - zeros[r][W+1]
    ru_zero = zeros[r-1][W+1] - zeros[r-1][c]
    ld_zero = zeros[H+1][c-1] - zeros[r][c-1]
    lu_zero = zeros[r-1][c-1]

    # 1つでも0が入っていれば0、そうでなければ区間積
    if rd_zero or ru_zero or ld_zero or lu_zero:
        print(0)
    else:
        res = rd * ru * ld * lu % MOD
        print(res)
