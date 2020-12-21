"""
・ちょっと時間かかったけど自力AC
・場合分け頑張る。
・check3まではわりとすぐ考えたんだけどWAで、check4のOK条件に気づくのに時間かかった。
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

S = [ord(s)-96 for s in input()]
N = len(S)

# 違う文字が1つでもあればそこをスワップすればいい
def check1(S):
    for i in range(1, N):
        if S[0] != S[i]:
            S[0], S[i] = S[i], S[0]
            return S
    return []

# 1文字を2文字に分ける
def check2(S):
    if S[0] != 1 and N < 20:
        tmp = S.pop()
        S.append(1)
        S.append(tmp-1)
        return S
    return []

# 2文字を1文字にまとめる
def check3(S):
    if S[0] <= 13 and N >= 2:
        tmp = S.pop()
        S.pop()
        S.append(tmp*2)
        return S
    return []

# 2文字を+-ゼロになるように動かす
def check4(S):
    if S[0] != 1 and S[0] != 26 and N >= 2:
        S[0] += 1
        S[1] -= 1
        return S
    return []

res = []
res.append(check1(S[:]))
res.append(check2(S[:]))
res.append(check3(S[:]))
res.append(check4(S[:]))
for i in range(4):
    if res[i]:
        ans = ''.join([chr(a+96) for a in res[i]])
        print(ans)
        exit()
NO()
