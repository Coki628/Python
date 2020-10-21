"""
・なんとか自力AC、WA叩きまくった。。
・実装、場合分け
・とりあえず基本方針は、ゴール1歩手前まで近づく、最後の1歩で偶奇の帳尻を合わせる、
　あとは操作回数がなくなるまで左右を繰り返す、みたいな感じ。
・でもこれだと最初からゴール地点にいる場合がコーナーケースになるので、そこは場合分け。
　まあちゃんと気付けたからよかったけど。。
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
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

t, gh, gw = MAP()

def check(h, w):
    return max(abs(h-gh), abs(w-gw))

# そもそも届かないはNG
ch = cw = 0
dist = check(ch, cw)
if dist > t:
    NO()
    exit()

ans = []
if gh == gw == 0:
    # 最初からゴール地点にいる場合は例外処理する
    if t == 1:
        NO()
        exit()
    elif t % 2 == 1:
        ans.append('^>')
        ans.append('v')
        ans.append('<')
        t -= 3
else:
    # ゴール1歩手前まで近づく
    while check(ch, cw) > 1:
        if abs(ch-gh) and abs(cw-gw):
            ans.append('^>')
            ch += 1
            cw += 1
            t -= 1
        elif abs(ch-gh):
            ans.append('^')
            ch += 1
            t -= 1
        else:
            ans.append('>')
            cw += 1
            t -= 1
    # 最後の1歩で偶奇の帳尻を合わせる
    if t % 2 == 1: 
        if abs(ch-gh) and abs(cw-gw):
            ans.append('^>')
            ch += 1
            cw += 1
            t -= 1
        elif abs(ch-gh):
            ans.append('^')
            ch += 1
            t -= 1
        else:
            ans.append('>')
            cw += 1
            t -= 1
    else:
        if abs(ch-gh) and abs(cw-gw):
            ans.append('^')
            ans.append('>')
            ch += 1
            cw += 1
            t -= 2
        elif abs(ch-gh):
            ans.append('^>')
            ans.append('<')
            ch += 1
            t -= 2
        else:
            ans.append('^>')
            ans.append('v')
            cw += 1
            t -= 2
# 操作回数がなくなるまで左右を繰り返す
while t:
    ans.append('<')
    ans.append('>')
    t -= 2

YES()
for s in ans:
    print(s)
