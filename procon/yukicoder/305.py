"""
・さくっと自力AC
・インタラクティブ問題
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

ans = [0] * 10
for i in range(10):
    s = str(i) * 10
    print(s, flush=1)
    x, res = input().split()
    if res == 'unlocked':
        exit()
    x = int(x)
    t = list(s)
    for j in range(10):
        t[j] = str((int(s[j])+1)%10)
        print(''.join(t), flush=1)
        x2, res = input().split()
        if res == 'unlocked':
            exit()
        x2 = int(x2)
        # j桁目をずらして減ったら、j桁目はiで合っていたことになる
        if x > x2:
            ans[j] = i
        t = list(s)
print(''.join(map(str, ans)), flush=1)
x, res = input().split()
