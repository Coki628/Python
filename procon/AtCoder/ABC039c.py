# -*- coding: utf-8 -*-

S = input()

# 白い鍵盤が連続する場所がミファかシドか判別
start = S.find('WW', 0)
end = S.find('WW', start+1)
key = end - start
# 位置が分かったら、それまでに出現した白鍵盤の数で出発点の音階を判定
cnt = S[:end].count('W')
# 愚直な場合分け…
if key == 7:
    if cnt == 6:
        print('Do')
    elif cnt == 5:
        print('Re')
    elif cnt == 4:
        print('Mi')
    exit()
elif key == 5:
    if cnt == 6:
        print('Fa')
    elif cnt == 5:
        print('So')
    elif cnt == 4:
        print('La')
    elif cnt == 3:
        print('Si')
    exit()