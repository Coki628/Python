# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc166/editorial.pdf
・自力ならず。
・1手先読みで貪欲、適切に場合分け
・まず、最初にsum(abc)が3以上あると、ほぼどうにでも最後まで行ける。
　これにそもそも気付けてなかったのが良くない。。
　1ずつの増減だから、考えたらそうだよね、行ったり来たりでよしなにできるよね。。
・sum(abc)が2の時は、例えば110→020の時に0,0を引かないようにクエリ先読みをする。
・それ以外は基本的に少なくなってる方を足してあげるでOK。
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

N, a, b, c = MAP()
C = {0: a, 1: b, 2: c}
S = []
for i in range(N):
    s = input()
    c1, c2 = s
    S.append((ord(c1)-65, ord(c2)-65))

ans = []
for i in range(N-1):
    cur1, cur2 = S[i]
    if C[cur1] == C[cur2] == 0:
        No()
        exit()
    elif C[cur1] == C[cur2] == 1 and C[3-cur1-cur2] == 0:
        if cur1 in S[i+1] and 3-cur1-cur2 in S[i+1]:
            ans.append(cur1)
            C[cur1] += 1
            C[cur2] -= 1
        else:
            ans.append(cur2)
            C[cur2] += 1
            C[cur1] -= 1
    elif C[cur1] <= C[cur2]:
        ans.append(cur1)
        C[cur1] += 1
        C[cur2] -= 1
    else:
        ans.append(cur2)
        C[cur2] += 1
        C[cur1] -= 1
cur1, cur2 = S[-1]
if C[cur1] == C[cur2] == 0:
    No()
    exit()
elif C[cur1] <= C[cur2]:
    ans.append(cur1)
    C[cur1] += 1
    C[cur2] -= 1
else:
    ans.append(cur2)
    C[cur2] += 1
    C[cur1] -= 1
Yes()
for i in range(N):
    print(chr(ans[i]+65))
