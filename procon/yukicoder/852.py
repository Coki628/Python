"""
・自力AC！
・区間、種類数、累積min、余事象
・平均を出す、みたいな問題だけど、実際はある文字が出現した区間数の総和が分かれば、
　そこから割ればいいので、ある文字が出現した区間数を各文字について求めることにする。
・ある文字が出現する区間より、1回も出現しない区間の方が数えやすそうなのでそっちでやる。
・出現しない区間は、今の位置を左端として、次にその文字が出現するところまで右端を伸ばせるので、
　次に出現する位置を前計算しておけばO(N*種類数)で求まる。
・あとは全体から引くとか最後に総区間数で割るとかやればOK。
・100万のN*種類数26の配列を確保したら最初pypyMLEした。。(315MB制限に対して626MB)
　accumulateがそこそこメモリ食うみたいのが前もあった気がしたので、
　累積minを普通にループ回したら無事pypyAC0.58秒メモリ282MB。
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

S = [ord(c)-97 for c in input()]
N = len(S)

nxt = list2d(26, N, N)
for i, c in enumerate(S):
    nxt[c][i] = i
for c in range(26):
    for i in range(N-2, -1, -1):
        nxt[c][i] = min(nxt[c][i], nxt[c][i+1])

total = N*(N+1) // 2
ans = 0
for c in range(26):
    cnt = 0
    for i in range(N):
        cnt += nxt[c][i] - i
    ans += total - cnt
ans /= total
print(ans)
