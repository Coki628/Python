"""
参考：https://twitter.com/long_ng_cp/status/1332696157991051266
・公式解の方針はいまいち頭に入って来ず、結構みんなメモ化再帰してたようなのでそっちで。
・よくよく話を聞いて考えてコード見て、やっと理解した。
・dp[i][j] := 左端位置がmod Nでiで、j回戦終わった時の勝者
　みたいなDP遷移を考える。ここまでは自分の考察でもちょっと考えたんだけど、
　うまい遷移を作れなかったから進めなかった。
・まずメモ化再帰では最終結果からトップダウンに1回戦まで潜っていく感じで進める。
　確かにこれだと1回戦開始前の人は確定しているので、順番に戻りでメモしていけばいけそう。
・で、遷移なんだけど、1回戦戻る毎に、今持ってる区間を半分にする感じなので、
　左端位置 + 2^(k-1) みたいにやると、欲しい位置に行ける。
　この時、そのままやると2^kはでかすぎるけど、全ての結果はmod Nで周期なので、
　この進む位置もmod Nしてやればいい。
・あとは普通にメモ化再帰で、1つ前で勝ってきた2人にチェック関数かけて
　今回勝つ方をメモに入れて戻す、ってやればOK。
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

def check(a, b):
    if a == b:
        return a
    if a == 'R' and b == 'S' \
            or a == 'S' and b == 'P' \
            or a == 'P' and b == 'R':
        return a
    else:
        return b

N, K = MAP()
S = input()

memo = list2d(N, K+1, '')
def rec(l, k):
    if k == 0:
        return S[l]
    if memo[l][k]:
        return memo[l][k]
    r = (l + pow(2, k-1, N)) % N
    a = rec(l, k-1)
    b = rec(r, k-1)
    res = check(a, b)
    memo[l][k] = res
    return res
ans = rec(0, K)
print(ans)
