# coding utf-8

import requests
from html.parser import HTMLParser
 
# HTMLParserを継承したクラスを定義する
class MyParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.flag1 = False # テーブルタグの場合のフラグ
        self.flag2 = False # スパンタグの場合のフラグ

    # 開始タグを扱うためのメソッド
    def handle_starttag(self, tag, attrs):
        # attrsリストの長さが1以上だったら処理する(こうしないとこの次で落ちる)
        if len(attrs) >= 1: 
            # inputタグで3番目の属性がname="fwp"の所を抽出
            if tag == "table" and attrs[0] == ("class", "infobox bordered"): # attrsはリスト内にタプルでkey,valueの構造になってる
                self.flag1 = True
                print("test1")
        if self.flag1 == True and tag == "span":
            self.flag2 = True
            print("test2")
    # 要素内用を扱うためのメソッド
    def handle_data(self, data): 
        if self.flag1:
            print(data)
            self.flag1 = False
            self.flag2 = False
            print("test3")
    # 終了タグを扱うためのメソッド
    def handle_endtag(self, tag):
        if tag == "table":
            self.flag1 = False




# パーサのインスタンスを生成してパースを実行
parser = MyParser()
parser.feed(requests.get("https://ja.wikipedia.org/wiki/西横浜駅").text)
# 改行のみ出力
