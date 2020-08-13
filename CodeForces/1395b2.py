"""
参考：https://codeforces.com/contest/1395/submission/89670312
・やっぱりみんな頭よかった。なんでこれ思いつかなかったんだ。。
・mod取りながら周期で横に動いて(つまり画面端→反対側へ)、最後だけ横に行かないで次の行へ、を繰り替えす。
　縦方向もmod取って周期でやれば、行も1つずつ下げてくだけでよしなにいける。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

W, H, sx, sy = MAP()
sx -= 1; sy -= 1

ans = []
nx = sx
for ny in range(sy, sy+H):
    ny %= H
    for i in range(W):
        ans.append((nx+1, ny+1))
        if i != W-1:
            nx = (nx + 1) % W

for x, y in ans:
    print(x, y)
