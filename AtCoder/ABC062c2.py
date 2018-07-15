# -*- coding: utf-8 -*-

H,W = map(int, input().split())

# T字で区切るパターンを縦横それぞれ試す
mnval = float("inf")
for i in range(1,H):
    # Tの横棒に当たる方
    tmpA = i * W
    # 縦2等分
    tmpB = (H-i) * (W // 2)
    # 残り
    tmpC = H*W - tmpA - tmpB
    mnval = min(mnval, abs(max(tmpA,tmpB,tmpC) - min(tmpA,tmpB,tmpC)))
for i in range(1,W):
    tmpA = i * H
    tmpB = (W-i) * (H // 2)
    tmpC = H*W - tmpA - tmpB
    mnval = min(mnval, abs(max(tmpA,tmpB,tmpC) - min(tmpA,tmpB,tmpC)))

# 縦並び横並びパターンをそれぞれ試す
if W >= 3:
    # 3等分
    tmpA = H * (W // 3)
    # 3等分と、その余った部分の半分
    tmpB = H * (W // 3 + ((W % 3) // 2))
    # 残り
    tmpC = H*W - tmpA - tmpB
    mnval = min(mnval, abs(max(tmpA,tmpB,tmpC) - min(tmpA,tmpB,tmpC)))
if H >= 3:
    tmpA = W * (H // 3)
    tmpB = W * (H // 3 + ((H % 3) // 2))
    tmpC = H*W - tmpA - tmpB
    mnval = min(mnval, abs(max(tmpA,tmpB,tmpC) - min(tmpA,tmpB,tmpC)))

print(mnval)