# -*- coding: utf-8 -*-

S = input()

cnt = 0
for i in range(1, len(S)):
    # ひとつ前と一致しない場所(白黒の境界)を数える
    if S[i] != S[i-1]:
        cnt += 1
print(cnt)