"""
・なんとか自力AC
・区間系、中央値
・大まかな方針はすぐ出たものの、細部を整えるのに時間かかった。
　こういう定番の区間系問題とかは手早く解きたいよなぁ。。
・要素が偶数の時、どっちの中央値使うかでどうしようと思ったけど、割とどっちでも良かったのかも。
　でも中央値との距離取って足し引きする時、どっちの中央値使ったかで距離変わっちゃうぽいけどどうなんだろう。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

# 例外処理
if len(set(A)) == 1:
    print(1)
    exit()

A.sort()
# まず右に全部寄せる場合の総和を出しておく
r = N // 2
lsm = rsm = 0
for i in range(r+1, N):
    rsm += A[i] - A[r]
for i in range(r-1, -1, -1):
    rsm += A[r] - A[i]

# 要素を1つずつ左に動かしていくのをN-1回試す
ans = INF
for i in range(N-1):
    # 左右それぞれの区間について中央値を出しておく
    l = i // 2
    r = (N+i) // 2
    # 今回の分を足し引き(中央値との距離)
    lsm += A[i] - A[l]
    rsm -= A[r] - A[i]
    ans = min(ans, lsm + rsm)
print(ans)
