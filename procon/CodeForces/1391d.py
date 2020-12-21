"""
・自力ならず。。
・グリッド、片方が極端に小さくなるからうまくいくDP
・縦横どっちか4以上は無理、にはなったけど、そこから貪欲で条件詰めようと頑張ってしまった。
・DPに落とす、が正解だった。縦横の小さい方は3以下になるので、
　直前3つの状態8通りを持ってDPを進めても、100万/3*8=266万くらいで間に合う。
・とはいえ、これ何か制約1秒なので、このdefaultdictに文字列として8通り持たせたやつはTLE。。
"""

import sys
from collections import defaultdict

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

H, W = MAP()
grid = [[]] * H
for i in range(H):
    grid[i] = list(input())

if H >= 4 and W >= 4:
    print(-1)
    exit()
if H == 1 or W == 1:
    print(0)
    exit()

if W > H:
    grid = list(zip(*grid))
    H, W = W, H

def calc(s, t):
    res = 0
    for i in range(len(s)):
        res += (s[i] != t[i])
    return res

dp = [defaultdict(lambda: INF) for i in range(H)]
cur = ''.join(grid[0])
if W == 2:
    for nxt in ('00', '01', '10', '11'):
        dp[0][nxt] = calc(cur, nxt)
    for i in range(1, H):
        cur = ''.join(grid[i])
        for k in dp[i-1].keys():
            if k in ('01', '10'):
                dp[i]['00'] = min(dp[i]['00'], dp[i-1][k] + calc(cur, '00'))
                dp[i]['11'] = min(dp[i]['11'], dp[i-1][k] + calc(cur, '11'))
            else:
                dp[i]['10'] = min(dp[i]['10'], dp[i-1][k] + calc(cur, '10'))
                dp[i]['01'] = min(dp[i]['01'], dp[i-1][k] + calc(cur, '01'))
else:
    for nxt in ('000', '001', '010', '011', '100', '101', '110', '111'):
        dp[0][nxt] = calc(cur, nxt)
    for i in range(1, H):
        cur = ''.join(grid[i])
        # to 000
        dp[i]['000'] = min(dp[i]['000'], dp[i-1]['101'] + calc(cur, '000'))
        dp[i]['000'] = min(dp[i]['000'], dp[i-1]['010'] + calc(cur, '000'))
        # to 001
        dp[i]['001'] = min(dp[i]['001'], dp[i-1]['011'] + calc(cur, '001'))
        dp[i]['001'] = min(dp[i]['001'], dp[i-1]['100'] + calc(cur, '001'))
        # to 010
        dp[i]['010'] = min(dp[i]['010'], dp[i-1]['111'] + calc(cur, '010'))
        dp[i]['010'] = min(dp[i]['010'], dp[i-1]['000'] + calc(cur, '010'))
        # to 011
        dp[i]['011'] = min(dp[i]['011'], dp[i-1]['110'] + calc(cur, '011'))
        dp[i]['011'] = min(dp[i]['011'], dp[i-1]['001'] + calc(cur, '011'))
        # to 100
        dp[i]['100'] = min(dp[i]['100'], dp[i-1]['001'] + calc(cur, '100'))
        dp[i]['100'] = min(dp[i]['100'], dp[i-1]['110'] + calc(cur, '100'))
        # to 101
        dp[i]['101'] = min(dp[i]['101'], dp[i-1]['111'] + calc(cur, '101'))
        dp[i]['101'] = min(dp[i]['101'], dp[i-1]['000'] + calc(cur, '101'))
        # to 110
        dp[i]['110'] = min(dp[i]['110'], dp[i-1]['100'] + calc(cur, '110'))
        dp[i]['110'] = min(dp[i]['110'], dp[i-1]['011'] + calc(cur, '110'))
        # to 111
        dp[i]['111'] = min(dp[i]['111'], dp[i-1]['101'] + calc(cur, '111'))
        dp[i]['111'] = min(dp[i]['111'], dp[i-1]['010'] + calc(cur, '111'))
ans = min(dp[-1].values())
print(ans)
