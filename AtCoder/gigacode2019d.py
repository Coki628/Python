# -*- coding: utf-8 -*-

"""
・C++でAC0.1秒。
・2次元累積和
・125^4のループは*1/4=約6100万になっててもつらくてpypyでもTLE。。
・と思ったら、試しにdef main()してみたら全然速くなってpypyAC1.2秒。
・前これやってもほぼ変化なくて使ってなかったんだけど、計算量厳しい時には差が出るみたいだな。。
・で、他の計算量つらいやつ(ABC147E)で試した結果、やっぱり却って遅くなった。ものによるみたいだ。
"""

def main():
    import sys
    from itertools import accumulate

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
    INF = 10 ** 18
    MOD = 10 ** 9 + 7

    def build_grid(H, W, intv, _type, space=True, padding=False):
        # 入力がスペース区切りかどうか
        if space:
            _input = lambda: input().split()
        else:
            _input = lambda: input()
        _list = lambda: list(map(_type, _input()))
        # 余白の有無
        if padding:
            offset = 1
        else:
            offset = 0
        grid = list2d(H+offset*2, W+offset*2, intv)
        for i in range(offset, H+offset):
            row = _list()
            for j in range(offset, W+offset):
                grid[i][j] = row[j-offset]
        return grid

    H, W, K, V = MAP()
    grid = build_grid(H, W, 0, int, padding=1)

    for i in range(H+2):
        grid[i] = list(accumulate(grid[i]))
    for j in range(W+2):
        for i in range(H+1):
            grid[i+1][j] += grid[i][j]

    ans = 0
    for h1 in range(1, H+1):
        for w1 in range(1, W+1):
            for h2 in range(h1, H+1):
                for w2 in range(w1, W+1):
                    land = grid[h2][w2] - grid[h1-1][w2] - grid[h2][w1-1] + grid[h1-1][w1-1]
                    square = (h2-h1+1) * (w2-w1+1)
                    price = land + square * K
                    if price <= V:
                        ans = max(ans, square)
    print(ans)
main()
