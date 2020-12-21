"""
・ここまで自力AC
・両方が好き、アリスだけ好き、ボブだけ好き、を分類してソート。
・結局2人ともK冊ないといけないので、「両方好き or アリスだけ+ボブだけ」で小さい方から選んでいく。
・最初の分類のelif b == 1をelseにしてどっちも好きじゃないを含んでしまったために1ペナ。。
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

N, K = MAP()
A = [INF]
B = [INF]
AB = [INF]
for i in range(N):
    t, a, b = MAP()
    if a == b == 1:
        AB.append(t)
    elif a == 1:
        A.append(t)
    elif b == 1:
        B.append(t)
A.sort(reverse=1)
B.sort(reverse=1)
AB.sort(reverse=1)

ans = 0
for i in range(K):
    if AB[-1] <= A[-1] + B[-1]:
        ans += AB.pop()
    else:
        ans += A.pop() + B.pop()
    if ans >= INF:
        print(-1)
        exit()
print(ans)
