"""
・さくっと自力AC
・日付時刻管理系
・python標準ライブラリにありがとう。
"""

import sys
from datetime import datetime, timedelta

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

h, m, s = input().split()
h = int(h)
m = int(m)

# 分に統一して差分を取る
a = 9 * 60
b = 0
if s.count('.'):
    b += int(s[-1]) * 6
    s = s[:-2]
if len(s) == 6:
    b += int(s[-2:]) * 60
    s = s[:-2]
else:
    b += int(s[-1:]) * 60
    s = s[:-1]
if s[-1] == '-':
    b = -b
diff = b - a

now = datetime(2020, 5, 19, h, m)
now += timedelta(minutes=diff)
ans = '{0}:{1}'.format(str(now.hour).zfill(2), str(now.minute).zfill(2))
print(ans)
