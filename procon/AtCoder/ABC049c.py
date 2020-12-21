# -*- coding: utf-8 -*-

S = input()
T = ""

words = []
words.append("dream")
words.append("dreamer")
words.append("erase")
words.append("eraser")
# 候補文字列を反転
for i in range(len(words)):
    words[i] = words[i][::-1]
# Sも反転
S = S[::-1]

while len(S) != 0:
    if S[:5] == words[0] or S[:5] == words[2]:
        S = S[5:]
    elif S[:6] == words[3]:
        S = S[6:]
    elif S[:7] == words[1]:
        S = S[7:]
    else:
        print("NO")
        exit()
print("YES")