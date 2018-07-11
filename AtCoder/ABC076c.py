# -*- coding: utf-8 -*-

S2 = input()
T = input()

# TをS2に当て込める位置の数ループ
for i in range(len(S2)-len(T)+1):
    # S2の後ろから、Tの文字数分切り出し
    tmpS = S2[len(S2)-len(T)-i:len(S2)-i]
    
    # Tの文字数分ループ
    for j in range(len(T)):
        # 対象の文字か?と一致しなければbreak
        if tmpS[j] != T[j] and tmpS[j] !=  "?":
            break
        # 最後まで条件を満たしたら出力準備
        if j == len(T) - 1:
            # 元のS2の該当箇所にTを当て込む
            S = S2[:len(S2)-len(T)-i] + T + S2[len(S2)-i:]
            # 残った?は全部a(辞書順最小がいいから)
            S = S.replace("?", "a")
            print(S)
            exit()
# 最後まで条件を満たす箇所がなければこっち
print("UNRESTORABLE")