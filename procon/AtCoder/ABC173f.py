"""
参考：https://www.youtube.com/watch?v=IMwigbYzLbI&feature=youtu.be
・自力ならず。。
・木、言い換え、式変形
・まず、グラフが木(森)の場合、連結成分の個数は「頂点数 - 辺数」と言い換えられる。
・これにより、sum(連結成分数)は、「sum(頂点数) - sum(辺数)」と式変形できる。
・ここまで来れば、後は実際のグラフの状態から、ある頂点・辺を取りうるLRの区間を数えればOK。
・そもそも1つ目の言い換えも出なかったけど、2つ目のsumの式変形も1つ目を言われただけだとピンと来なかった。
　解説動画で説明されてやっとハッとした。前もこのsumの式変形出なかったんだよなぁどっかで。
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

N = INT()
vcnt = 0
for i in range(1, N+1):
    # 頂点iを含む(左端の数 * 右端の数)
    vcnt += i * (N+1-i)
ecnt = 0
for i in range(N-1):
    a, b = MAP()
    if a > b:
        a, b = b, a
    # 頂点aを含む左端の数 * 頂点bを含む右端の数
    ecnt += a * (N+1-b)
ans = vcnt - ecnt
print(ans)
