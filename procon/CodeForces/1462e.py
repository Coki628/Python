"""
・なんとか自力AC。。
・数え上げ、最小を固定
・なんとなく、3つの問題だと真ん中を固定したくなって、うまくいかなくて時間かかった。
・下を固定すると、6通り(詳細はソース内コメント)に場合分けして重複なく数えられる。
・でもこの方針だとE2の方は解けない。。。
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

def nC2(n):
    return n*(n-1)//2

def nC3(n):
    return n*(n-1)*(n-2)//6

for _ in range(INT()):
    N = INT()
    A = LIST()

    C = [0] * (N+7)
    for a in A:
        C[a] += 1
    cnt = 0
    for i in range(1, N+1):
        # iを3つ使う
        cnt += nC3(C[i])
        # iを2つ,i+1を1つ使う + iを1つ,i+1を2つ使う
        cnt += nC2(C[i])*C[i+1] + C[i]*nC2(C[i+1])
        # iを2つ,i+2を1つ使う + iを1つ,i+2を2つ使う
        cnt += nC2(C[i])*C[i+2] + C[i]*nC2(C[i+2])
        # i,i+1,i+2を1つずつ使う
        cnt += C[i]*C[i+1]*C[i+2]
    print(cnt)
