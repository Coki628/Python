"""
参考：https://twitter.com/kichi2004_/status/1305139368596627457
・自力ならず。。
・kichiさんを参考に、普通に半分入れ替えてうまくいってない箇所はスワップするようにした。
　とりあえず乱択は入れないで素直に前からループ回して探すようにしたが、これでAC。
・ちなみに探す時乱択やるようにしたらそれはTLE。。。
"""

import sys
from collections import Counter

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

N = INT()
A = LIST()
B = LIST()

C1 = Counter(A)
C2 = Counter(B)
for k, v in C1.items():
    if v > N-C2[k]:
        No()
        exit()

# ここまででWAがあと2つだけになる
ans = B[ceil(N, 2):] + B[:ceil(N, 2)]

def check(A, B):
    for i in range(N):
        # 重複部分を見つけたらそこのindexを返す
        if A[i] == B[i]:
            return i
    return -1

def swap(x, A, B):
    for i in range(N):
        # スワップしてどちらのペアも値が被らないような場所でやる
        if A[x] != B[i] and A[i] != B[x]:
            B[x], B[i] = B[i], B[x]
            return True
    return False

while 1:
    res = check(A, ans)
    if res == -1:
        Yes()
        print(*ans)
        break
    else:
        if not swap(res, A, ans):
            No()
            break
