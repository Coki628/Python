"""
・終了5分後くらいでAC、、悔しい。。。
・絶対2重ループで貪欲に何かだって思ってその何かがなかなか出ず。。
・結局、最終的に合わせる値としてはどこかの区間和しかない訳で、それを前から全探索すればいい。
　合わせる値を決め打ちは早い段階から考えたんだけど、max(A)が10万くらいあったので、
　全部の値でやったら内側でループ回すの無理だぁくぅぅってなってた。。
・先に、小さい値から見ていけばよさそうみたいな謎貪欲に走ってしまったのも敗因。。
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

for _ in range(INT()):
    N = INT()
    A = LIST()

    ans = N - 1
    for x in range(1, N):
        sm = sum(A[:x])
        cur = 0
        cnt = x - 1
        for i in range(x, N):
            cur += A[i]
            if cur > sm:
                break
            elif cur == sm:
                cur = 0
            else:
                cnt += 1
        if cur == 0:
            ans = min(ans, cnt)
    print(ans)
