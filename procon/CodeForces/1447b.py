"""
・実験用。
・ここから、切り替える位置の数が偶数個であれば、基本的にどう変更するのもいけそう、
　という様子が見えてくる。。
・マス間の辺全部を切り替え対象としてビットに当てはめるので、実装ちょっとだるいな、ってためらった。
　でも書いてみたら思ったほど面倒くもなかったので、さっさと書いてみればよかったね。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = LIST()

    # 位置kを操作するかしないか、で全状態をビット全探索
    for bit in range(1<<(H*(W-1)+(H-1)*W)):
        ans = list2d(H, W, 0)
        k = 0
        for i in range(H):
            for j in range(W-1):
                if bit & 1<<k:
                    # この位置を切り替える
                    ans[i][j] ^= 1
                    ans[i][j+1] ^= 1
                k += 1
        for j in range(W):
            for i in range(H-1):
                if bit & 1<<k:
                    ans[i][j] ^= 1
                    ans[i+1][j] ^= 1
                k += 1
        print()
