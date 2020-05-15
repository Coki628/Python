"""
・ここまで自力AC
・優先度付きキューでシミュレーション
"""

import sys
from heapq import heappush, heappop

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()

    # 長さの降順、左端位置の昇順を優先順位にする
    que = [(-N, 0)]
    ans = [0] * N
    for i in range(N):
        ln, l = heappop(que)
        ln = -ln
        r = l + ln
        m = ceil(l+r, 2) - 1
        ans[m] = i + 1
        # この位置を操作した後2つに分かれた区間をキューに詰める
        heappush(que, (-(m-l), l))
        heappush(que, (-(r-(m+1)), m+1))
    print(*ans)
