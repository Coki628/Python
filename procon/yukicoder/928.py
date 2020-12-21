"""
・自力ならず。。
・場合分け、範囲を絞って全探索
・方針がダメだった。。切り捨ててる分、境界付近の値がブレるから、にぶたん+境界付近全探索で頑張ってた。
・元の値段xが十分大きくなると、容器代rがどんな値であれp,qの差が覆らなくなる。
　なのでp,qの大小で場合分けして、p>qなら大きい値は常にNG、p<qなら大きい値は常にOKとすれば、
　値が小さいところを全探索でいける。どこまで見るかは、公式解が色々式変形して証明してくれているが、
　100*max(容器代)=100万くらい見ればOK。(この辺は自分でやるなら適当にやることになりそうだが。。)
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

def check(x):
    a = (100+p) * x // 100
    b = (100+q) * x // 100 + r
    return a < b

p, q, r = MAP()
MX = 10**9
N = 10**6

if p == q:
    if r == 0:
        print(0)
    else:
        print(MX)
elif p > q:
    ans = 0
    for i in range(1, N+1):
        if check(i):
            ans += 1
    print(ans)
else:
    ans = 0
    for i in range(1, N+1):
        if check(i):
            ans += 1
    ans += MX - N
    print(ans)
