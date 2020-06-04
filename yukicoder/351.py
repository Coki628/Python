"""
参考：https://yukicoder.me/problems/no/351/editorial
　　　https://kimiyuki.net/writeup/algo/yukicoder/351/
・自力ならず。。
・逆操作
・行列が複雑に入れ替わって途方もなく感じてしまったが、逆操作すれば元位置を辿れた。
　逆から動かすのは典型なのに、どうして気づかなかったかなぁ。。
・しかも解法分かってもWAが取れず。よく考えたら、「行がスライドすれば列番号が動く、
　列がスライドすれば行番号が動く」なんだよね。逆やってたね。。そこ直してAC。
・100万クエリでpythonどうかなーと思ったけど、間に合ってAC1.4秒。
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

H, W = MAP()
N = INT()
Qs = []
for i in range(N):
    s, k = input().split()
    Qs.append((s, int(k)))

ch = cw = 0
for s, k in Qs[::-1]:
    if s == 'R' and k == ch:
        cw = (cw-1) % W
    if s == 'C' and k == cw:
        ch = (ch-1) % H

if (ch+cw) % 2 == 0:
    print('white')
else:
    print('black')
