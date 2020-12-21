"""
参考：https://codeforces.com/blog/entry/83761
・自力ならず。。
・端的に言えば、端を1個増やす、全体のやる、反対端を1個増やす、の3ステップで終了。
・2回目で全体をやれば、ちょうど1回目にやった端を除いた部分が反対端に来るので、
　その反対端にも1回目と同じことをやればいい。
・操作回数30回制限はブラフだったなー。。アルファベット全部分くらいでなんかゴニョゴニョやるのかと
　思ってしまった。。中身の文字列一切関係ないし、まじでこどふぉっぽいギャグ問ね。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

S = input()
N = len(S)

ans = []
ans.append(('R', N-1))
ans.append(('L', N))
ans.append(('L', 2))

print(len(ans))
for a in ans:
    print(*a)
