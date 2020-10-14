"""
参考：https://drken1215.hatenablog.com/entry/2020/10/05/133100
　　　https://noimin.hatenablog.com/entry/2020/10/04/192418
・区間DP
・でも厄介なのは区間DPよりも、不正入力としんどい判定パートな気がする。。
・とりあえず区間DPパートは、みんな何か1次元の見たことない形にしてたけど、
　自分でやったことのある、2次元配列でN^3の区間DPの形に落とし込んだ。
　多分本質的にやってることは同じだと思う、多分。。
・判定パートは丁寧に考えてまとめた。これでダメならもう寝かせようかなと思ったら通った。
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

N = INT()
S = set()
S1 = set()
S2 = set()
used = [-1] * (N*2)
for i in range(N):
    a, b = MAP()
    a -= 1; b -= 1
    # 明らかな不正を弾く
    if a != -2 and b != -2 and a >= b or a in S or b in S:
        No()
        exit()
    if a != -2:
        S.add(a)
        S1.add(i)
        used[a] = i
    if b != -2:
        S.add(b)        
        S2.add(i)
        used[b] = i
N *= 2

def check(l, r):
    d = r - l
    for i in range(d//2):
        fr = used[l+i]
        to = used[l+d//2+i]
        # fromとtoが一致してればOK
        if fr == to:
            continue
        # fromが未確定でtoがfrom側で未出現ならここに当てればいいのでOK
        if fr == -1 and to not in S1:
            continue
        # toについても同じ
        if to == -1 and fr not in S2:
            continue
        return False
    return True

# dp[l][r] := 区間[l,r)でグループを形成できるかどうか
dp = list2d(N+1, N+1, False)
for i in range(N-1):
    dp[i][i+2] = check(i, i+2)
# 偶数長の区間だけやる
for d in range(4, N+1, 2):
    for l in range(N+1-d):
        r = l + d
        for m in range(l+1, r):
            dp[l][r] |= dp[l][m] & dp[m][r]
        # この[l,r)を1つの区間としてチェック
        dp[l][r] |= check(l, r)
if dp[0][N]:
    Yes()
else:
    No()
