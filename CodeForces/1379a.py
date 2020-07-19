"""
・なんとか自力AC。Aなのにむずかった。
・まず?なしでOKなケースとNGなケースを処理。以下、?を使う前提でいく。
・ある位置から?ありで目的の文字列を作れるか、で全位置試す。
　うまくいった場所には、その位置で使った?以外を適当な文字に置き換えた状態で、
　余計なところに目的の文字列が発生していないかをチェック。
・最悪ケースでは文字列スライスを無視してもN^2のループなので通るか心配だったけど大丈夫だった。
　テストケース5000,N<=50でNが小さいからよかったんだろう多分。。
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

T = 'abacaba'
M = len(T)

def check1(S):
    for i in range(M):
        if S[i] != T[i]:
            return False
    return True

def check2(S):
    for i in range(M):
        if S[i] != '?' and S[i] != T[i]:
            return False
    return True

for _ in range(INT()):
    N = INT()
    S = input()

    # まず?なしでOKなケースとNGなケースを処理
    cnt = 0
    ans = ''
    idx = -1
    for i in range(N-M+1):
        if check1(S[i:i+M]):
            cnt += 1
            idx = i
    if cnt >= 2:
        No()
        continue
    elif cnt == 1:
        Yes()
        ans = S[:idx] + T + S[idx+M:]
        ans = ans.replace('?', 'x')
        print(ans)
        continue

    # ここから?を使うのをやる
    idx = -1
    ok = False
    for i in range(N-M+1):
        # うまくいった場所には、余計なところに目的の文字列が発生していないかをチェック
        if check2(S[i:i+M]):
            idx = i
            ans = S[:idx] + T + S[idx+M:]
            ans = ans.replace('?', 'x')
            cnt = 0
            for j in range(N-M+1):
                # 愚直に全箇所見て間に合う
                if check1(ans[j:j+M]):
                    cnt += 1
            if cnt >= 2:
                continue
            elif cnt == 1:
                ok = True
                break
            else:
                raise Exception
    if ok:
        Yes()
        print(ans)
    else:
        No()
