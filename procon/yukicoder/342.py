"""
・さくっと自力AC
・制約は小さいので、2重ループ作って愚直に調べればいい。と思ったけどこのwhileのループ多分線形だ。
・「〇〇ｗｗ」の組をとりあえず全部〇〇とｗの長さで持っておいて、最終的に最大の所だけ出力させる、って感じにした。
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

S = input()
N = len(S)

i = 0
ans = []
mx = 0
while i < N-1:
    if S[i] != 'ｗ' and S[i+1] == 'ｗ':
        j = i
        while j >= 0 and S[j] != 'ｗ':
            j -= 1
        s = S[j+1:i+1]
        j = i + 1
        while j < N and S[j] == 'ｗ':
            j += 1
        ln = j - i - 1
        ans.append((s, ln))
        mx = max(mx, ln)
        i = j
    else:
        i += 1

for s, ln in ans:
    if ln == mx:
        print(s)
