"""
参考：https://yukicoder.me/problems/no/1034/editorial
　　　https://qiita.com/c-yan/items/9df514cda87727acc0d2#b-1034-%E3%83%86%E3%82%B9%E3%82%BF%E3%83%BC%E3%81%AE%E3%81%B5%E3%81%A3%E3%81%B4%E3%83%BC%E3%81%95%E3%82%93-
・自力ならず。。
・グリッド
・10億*10億グリッドは線形時間も無理なので考察O(1)をする必要がある。
・内容はソース内コメント参照。これは出なかったなー。。
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

for _ in range(INT()):
    N, i, j = MAP()

    # (i, j)が何周目か(0-indexed)
    turn = min(i, j, N-i-1, N-j-1)
    # turn周目の1辺の長さ
    N2 = N - turn*2
    # turn周目に入るまでにかかるコスト
    cost = N**2 - N2**2
    # turn周目右下の座標
    p = (N-turn-1, N-turn-1)
    # (i, j)との差分
    diff = abs(sum(p) - (i+j))
    # 左下半分か右上半分かで場合分けして足し引き
    if i <= j:
        cost += (N2-1)*2 - diff
    else:
        cost += (N2-1)*2 + diff
    print(cost)
