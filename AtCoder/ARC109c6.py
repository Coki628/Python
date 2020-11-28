"""
参考：https://twitter.com/shiro537/status/1332711787540541443
　　　https://twitter.com/kyopro_friends/status/1332712906778050562
・別解。こっちの方針で解いた人も結構いるみたい。
・ダブリング
・ダブリングもね、確かに考えたんだよ。いい感じにできなかったけど。。
・正解は、ダブリングで和とかmaxも取りたい時みたいな感じでやる。
　(文章化しづらい。詳細はソースコード参照。)
・今回は構築したダブリング配列で何かを高速に取るとかではなく、
　ダブリング配列に入ってる値そのものが答えになる。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def check(a, b):
    if a == b:
        return a
    if a == 'R' and b == 'S' \
            or a == 'S' and b == 'P' \
            or a == 'P' and b == 'R':
        return a
    else:
        return b

def doubling(MAX, A):
    """ ダブリング """

    N = len(A)
    nxt = list2d(MAX, N, -1)
    win = list2d(MAX, N, '')
    for i, a in enumerate(A):
        nxt[0][i] = (i+1) % N
        win[0][i] = a
    for k in range(1, MAX):
        for i in range(N):
            nxt[k][i] = nxt[k-1][nxt[k-1][i]]
            # ここで和とかmaxが欲しい時と同じ遷移をやる
            win[k][i] = check(win[k-1][i], win[k-1][nxt[k-1][i]])
    return (nxt, win)

N, K = MAP()
S = input()

_, res = doubling(K+1, list(S))
ans = res[K][0]
print(ans)
