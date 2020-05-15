"""
参考：https://babuyade.hatenablog.com/entry/2020/05/12/131309
・場合分け。やっぱりもっと簡単にできた。
　全部1か2で埋めて残りを1箇所に詰める方針にするとよかった。。
・総和Nと数列の長さKの偶奇で4通りの場合を考える。
・総和奇数、長さ偶数はどうやっても無理、
　総和偶数、長さ奇数はパリティを0にしないといけないので2で埋めるのにNが足りるかチェック、
　それ以外2つは、パリティが1でいいので1で埋めるのにNが足りるかチェック。　
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()

    if N % 2 == 1 and K % 2 == 0:
        NO()
        continue
    elif N % 2 == 0 and K % 2 == 1:
        if N < K*2:
            NO()
            continue
        ans = [2] * K
        ans[0] += N - K*2
    else:
        if N < K:
            NO()
            continue
        ans = [1] * K
        ans[0] += N - K
    YES()
    print(*ans)
