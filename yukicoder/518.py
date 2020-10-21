"""
・自力AC
・実装おつ系
・例外処理頑張って素直にやったけど、先に定型に合わせて置換しちゃえば楽だったぽい。。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

D = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000 }
N = INT()
A = input().split()

sm = 0
for i in range(N):
    s = A[i]
    cnt = D[s[0]]
    for j in range(len(s)-1):
        # 4,9は例外処理
        if s[j:j+2] in ['IV', 'IX', 'XL', 'XC', 'CD', 'CM']:
            cnt += D[s[j+1]] - D[s[j]]*2
        else:
            cnt += D[s[j+1]]
    sm += cnt

if sm >= 4000:
    print('ERROR')
    exit()

ans = ''
cur, sm = divmod(sm, 1000)
ans += 'M' * cur
cur, sm = divmod(sm, 100)
# 4,9は例外処理
if cur == 4:
    ans += 'CD'
elif cur == 9:
    ans += 'CM'
else:
    d, m = divmod(cur, 5)
    ans += 'D'*d + 'C'*m
cur, sm = divmod(sm, 10)
if cur == 4:
    ans += 'XL'
elif cur == 9:
    ans += 'XC'
else:
    d, m = divmod(cur, 5)
    ans += 'L'*d + 'X'*m
cur, sm = divmod(sm, 1)
if cur == 4:
    ans += 'IV'
elif cur == 9:
    ans += 'IX'
else:
    d, m = divmod(cur, 5)
    ans += 'V'*d + 'I'*m
print(ans)
