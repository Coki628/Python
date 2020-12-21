# -*- coding: utf-8 -*-

S = input()

# 例外処理
if len(S) == 1:
    print(1)
    exit()

s = []
t = []
mx = tmp = 1
# 最後の文字をループ内で判定したいのでダミーを足しておく
S += 'a'
for i in range(1, len(S)):
    if S[i] == S[i-1]:
        tmp += 1
    # 同じ文字が連続する最大箇所を探す
    elif tmp >= 2:
        if tmp > mx:
            mx = tmp
            # 始点と終点の情報と取っておく
            s = [i-mx]
            t = [i]
        # 同率なら両方残す
        elif tmp == mx:
            s.append(i-mx)
            t.append(i)
        tmp = 1
# ダミーを取り除く
S = S[:-1]

# 例外処理
if mx == 1 and len(S) >= 3:
    print(2)
    exit()

# 一番長く続いた箇所(複数の場合もある)について調べる
for i in range(len(s)):
    # 連続箇所の両端の先に文字があるかどうか
    if s[i]-1 >= 0 and t[i] < len(S):
        mx += 1
        break
print(mx)
