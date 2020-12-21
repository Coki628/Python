"""
参考：https://codeforces.com/blog/entry/79107
　　　https://codeforces.com/contest/1370/submission/84494808
・自力ならず。
・0,1の数が合わなければNG、あと既に揃ってる所はいらない。これで0,1同数の数列が作れる。
　(この方が簡単になるけど、別にそのままでもできる)
・ここから0,1を消すのと1,0を消すのを両方見て、一番回数にかかるところが答え。
・周期なので、2周すると両方の最大は一致する(多分)ので片方調べるだけでもいい。
・文字列の連結100万回あるとTLEで、配列にしたらAC0.4秒。
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

N = INT()
S = input()
T = input()

if S.count('0') != T.count('0'):
    print(-1)
    exit()

S2 = []
for i in range(N):
    if S[i] != T[i]:
        S2.append(S[i])

ans = 0
cnt0 = cnt1 = 0
for s in S2:
    if s == '0':
        cnt0 += 1
        cnt1 -= 1
    else:
        cnt0 -= 1
        cnt1 += 1
    cnt0 = max(cnt0, 0)
    cnt1 = max(cnt1, 0)
    ans = max(ans, cnt0, cnt1)
print(ans)
