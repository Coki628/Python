# -*- coding: utf-8 -*-

s = input()

def typeStr(ans, str1):
    if str1 == "B":
        # 文字列が空ではない確認
        if ans != "":
            # 残っている時のみ削除を行う(最後の1文字以外を切り取る)
            ans = ans[0:-1]
    else:
        ans += str1
    return ans

ans = ""
for i in range(len(s)):
    ans = typeStr(ans, s[i])

print(ans)