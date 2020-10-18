"""
・なんとかここまで自力AC！
・後ろから決めていく。
・1,2,3の場所でそれぞれいい置き方のパターンがある。(自分ノート参照)
・基本はそれに従って置く。2のやつは1のところとペアになる感じに置く。
　でも3を被せられると、1は2とペアにできなくなるからそのへんも管理する。
・条件足りてるかめちゃめちゃ自信なかったけど、無事通ってよかった。
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

N = INT()
A = LIST()

started = False
currow = N
has1 = []
ans = []
prevcol = -1
for i in range(N-1, -1, -1):
    if A[i] == 1:
        ans.append((currow, i+1))
        has1.append(currow)
        prevcol = i+1
        currow -= 1

    elif A[i] == 2:
        if not has1:
            print(-1)
            exit()
        r = has1.pop()
        ans.append((r, i+1))
        prevcol = i+1

    elif A[i] == 3:
        if prevcol == -1:
            print(-1)
            exit()
        if has1 and has1[-1] == currow+1:
            has1.pop()
        ans.append((currow, prevcol))
        ans.append((currow, i+1))
        prevcol = i+1
        currow -= 1

print(len(ans))
for a in ans[::-1]:
    print(*a)
