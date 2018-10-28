# -*- coding: utf-8 -*-

S = input()

idx = 0
ans = 0
for i in range(len(S)):
    # +があったらそこまでの式に0があったか調べる
    if S[i] == '+':
        if S[idx:i].find('0') == -1:
            ans += 1
        # 開始位置をずらす
        idx = i + 1
    # 最後は+ないので別処理
    if i == len(S)-1:
        if S[idx:].find('0') == -1:
            ans += 1
print(ans)