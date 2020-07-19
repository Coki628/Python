"""
・なんとか自力AC。
・式をよく観察すると、nは0以外何でもいいのでn*aの部分でaの倍数はどうにでも作れる、となる。
・あとはそれではみ出た部分を+b-cで埋められるかで、bもcも[l,r]の範囲内なら何でもいいので、
　b-cで常に0は作れる、+-1以上は[l,r]の範囲がそれ以上大きければ作れる、となる。
・以上より、n*aが固まればb,cの目指すところは一意になるのでaを全探索。
・aを固定した時、差分として必要になるのは、m%aかその反対のやつ(pythonなら-m%a)のどちらか。
　両方試して、b,cが[l,r]の範囲に収まるやつがあったらそれを使えばいい。
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
    l, r, m = MAP()

    for a in range(l, r+1):
        need = m % a
        n = m // a
        # サンプルでも優しく説明されてるけど、n=0はダメ
        if n != 0:
            b = r
            c = r - need
            if c >= l:
                print(a, b, c)
                break
        need = -m % a
        n = m // a + 1
        c = r
        b = r - need
        if b >= l:
            print(a, b, c)
            break
