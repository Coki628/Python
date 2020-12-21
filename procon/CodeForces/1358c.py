"""
・なんとか自力AC
・等差数列の和
・色々悩んだ。取りうる値は最大から最小までの全てで、一番上を通るルートが最大、一番下を通るルートが最小、
　が分かって、そこから更に考えて、1マス移動する毎に両方の差分は12333333321みたいになると分かった。
・で、縦横の移動回数から、等差数列部分と同値部分に分けて計算した。
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

def get_sum(a, d, n):
    """ 等差数列の和：(初項a, 公差d, 項数n) """
    return (2*a + (n-1)*d) * n // 2

for _ in range(INT()):
    x1, y1, x2, y2 = MAP()

    cntx = x2 - x1
    cnty = y2 - y1

    mn = min(cntx, cnty)
    # 等差数列部分*2(最初と最後) + 同値部分(一番離れた距離*(総移動回数-1-等差数列部分の回数)) + 1
    ans = get_sum(1, 1, mn) * 2 + mn * (cntx+cnty-1-mn*2) + 1
    print(ans)
