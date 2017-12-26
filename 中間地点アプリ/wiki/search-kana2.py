# coding utf-8

import requests
from html.parser import HTMLParser
import MySQLdb
import jaconv
"""
駅名読み仮名取得スクリプト2
　DBから取得した駅名に対して、wikiから目的のタグを読み込み、仮名の情報でDBを更新する
"""
# HTMLParserを継承したクラスを定義する
class MyParser(HTMLParser):

    # コンストラクタ
    def __init__(self):
        HTMLParser.__init__(self)
        self.isKana = False # 仮名の場合のフラグ
        self.isDone = False # 処理済フラグ

    # 開始タグを扱うためのメソッド
    def handle_starttag(self, tag, attrs):
        # まだ未処理の場合のみ対象とする
        if self.isDone == False:
            # attrsリストの長さが1以上だったら処理する(こうしないとこの次で落ちる)
            if len(attrs) >= 1: 
                # spanタグで1番目の属性がstyle="font-size:120%;"の所を抽出
                if tag == "span" and attrs[0] == ("style", "font-size:120%;"): # attrsはリスト内にタプルでkey,valueの構造になってる
                    self.isKana = True
                # 曖昧さ回避ページに飛んだ時の処理
                elif tag == "a" and len(attrs) >= 2:
                    # 現在処理中の都道府県名と駅名を取得
                    global pref
                    global st_name
                    # 一致する"駅名 (都道府県名)"があれば駅名に格納                  
                    if attrs[1] == ("title", st_name + " (" + pref + ")"):
                        st_name = attrs[1][1]
                        self.isDone = True
                    
    # 要素内容を扱うためのメソッド
    def handle_data(self, data): 
        if self.isKana:
            # 取得した仮名をメイン処理で宣言した変数kanaに格納
            global kana
            # カタカナを平仮名にする処理を入れる
            kana = jaconv.kata2hira(data)
            print(kana, end = " ")
            self.isKana = False
            self.isDone = True

    # 終了タグを扱うメソッド
    def handle_endtag(self, tag):
        # htmlタグの終了時に処理済フラグを戻す
        if tag == "html":
            self.isDone = False

# パーサのインスタンスを生成
parser = MyParser()
# 取得した仮名を格納する変数
kana = ""
# 処理中の駅名を格納する変数
st_name = ""
# 処理中の都道府県名を格納する変数
pref = ""

# データベース接続とカーソル生成
conn = MySQLdb.connect(
    host='localhost', user='Coki', passwd='0628', db='app', charset='utf8')
# 引数指定で結果をタプルではなく辞書型で取得
cursor = conn.cursor(MySQLdb.cursors.DictCursor)

# エラー処理（例外処理）
try:
    # 取得したレコードを全件ループ
#    cursor.execute("SELECT * FROM station WHERE id BETWEEN 66 AND 78 ORDER BY id ASC")
#    cursor.execute("SELECT * FROM station WHERE pref_cd IN (47) ORDER BY id ASC")
    cursor.execute("SELECT * FROM station WHERE pref_cd IN (47) AND st_kana IS NULL ORDER BY id ASC")
    for row in cursor:
        if row["st_name"] != "" and row["st_name"] is not None:
            print(row["st_name"], end = " ")
            # 都道府県コードから処理中の都道府県名を取得
            cursor.execute("SELECT * FROM pref WHERE pref_cd = %s", (row["pref_cd"],)) # 要素1個のタプルは後ろに,つけないとダメらしい
            pref = cursor.fetchone()["pref_name"]
            # 処理中の駅名を取得
            st_name = row["st_name"] + "駅"
            # 駅名を使ってリクエストを投げて結果を取得
            parser.feed(requests.get("https://ja.wikipedia.org/wiki/" + st_name).text)
            # 仮名が取得できたらDB更新
            if kana != "":
                cursor.execute("UPDATE station SET st_kana = %s WHERE id = %s", (kana, row["id"]))
                # 保存を実行（忘れると保存されないので注意）
                conn.commit()
                # 取得した仮名を初期化
                kana = ""
                print("更新完了")
            # 駅名が書き換わっていたら再度リクエストする
            elif st_name != row["st_name"] + "駅":
                # 駅名を使ってリクエストを投げて結果を取得
                parser.feed(requests.get("https://ja.wikipedia.org/wiki/" + st_name).text)
                # 仮名が取得できたらDB更新
                if kana != "":
                    cursor.execute("UPDATE station SET st_kana = %s WHERE id = %s", (kana, row["id"]))
                    # 保存を実行（忘れると保存されないので注意）
                    conn.commit()
                    # 取得した仮名を初期化
                    kana = ""
                    print("更新完了")
            else:
                print("仮名が見つかりません")

except MySQLdb.Error as e:
    print('MySQLdb.Error: ', e)

# 接続を閉じる
conn.close()

print("処理終了")


