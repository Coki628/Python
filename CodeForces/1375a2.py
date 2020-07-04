"""
参考：https://codeforces.com/contest/1375/submission/85981558
・かっつさんのツイートから。
・乱択
・これで通るんだな。。
・計算量はN最大99でテストケース500個なんだけど、多分1ケースにつき100回くらいやれば合うんだと思う。
　(ちょうど正負半分ずつのパターンは全体からしてもかなり多いはずなので、多分。)
"""

import sys
from random import randrange

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

for _ in range(INT()):
    N = INT()
    A = LIST()

    while 1:
        ucnt = dcnt = 0
        for i in range(N):
            if randrange(0, 2):
                A[i] = -A[i]
        for i in range(N-1):
            if A[i] <= A[i+1]:
                ucnt += 1
            if A[i] >= A[i+1]:
                dcnt += 1
        if ucnt >= N//2 and dcnt >= N//2:
            break
    print(*A)
