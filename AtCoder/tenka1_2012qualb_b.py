"""
・自力AC！絶対考慮漏れコーナーありそうと思って投げたら1発で通ったので嬉しい。
・文字列、実装おつ系
・色々な可能性を考慮しながら実装する。詳細はソース内コメント。
"""

import sys
from string import ascii_lowercase, ascii_uppercase, digits

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

# 単語は英小文字から始まる必要があるので、まず英小文字がないのはNGとみなす
ok = False
for c in ascii_lowercase:
    if c in S:
        ok = True
        break
if not ok:
    print(S)
    exit()

# 左端、右端のアンスコはメインの判定とは別に処理
unscol = 0
unscor = 0
for i in range(N):
    if S[i] == '_':
        unscol += 1
    else:
        l = i
        break
for i in range(N-1, -1, -1):
    if S[i] == '_':
        unscor += 1
    else:
        r = i + 1
        break
S = S[l:r]
N = len(S)

# スネークケース判定
def is_snake(S):
    words = S.split('_')
    for word in words:
        # アンスコ連続はここで弾く
        if not word:
            return False
        # 単語の先頭は英小文字必須
        if word[0] not in ascii_lowercase:
            return False
        for c in word:
            # 先頭以外は英小文字か数字(英大文字を弾く)
            if c not in ascii_lowercase + digits:
                return False
    return True

# キャメルケース判定
def is_camel(S):
    # 最初の単語の先頭は英小文字必須
    if S[0] not in ascii_lowercase:
        return False
    # アンスコが含まれてたらダメ
    if S.count('_'):
        return False
    return True

# スネークケースだったらキャメルケースを構築
if is_snake(S):
    words = S.split('_')
    ans = ''
    for i, word in enumerate(words):
        # 最初の単語以外は先頭大文字にする
        if i == 0:
            ans += word
        else:
            ans += word.capitalize()
    ans = '_'*unscol + ans + '_'*unscor
    print(ans)
    exit()

# キャメルケースだったらスネークケースを構築
if is_camel(S):
    words = []
    word = ''
    # ランレングス圧縮の要領で、大文字が来たら区切る、をやる
    for i in range(N):
        if S[i] in ascii_uppercase:
            words.append(word.lower())
            word = S[i]
        else:
            word += S[i]
    words.append(word.lower())
    ans = '_'.join(words)
    ans = '_'*unscol + ans + '_'*unscor
    print(ans)
    exit()

# どちらも無理ならそのまま
ans = S
ans = '_'*unscol + ans + '_'*unscor
print(ans)
