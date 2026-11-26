# daily-settlement-tool
AI推定日次決算ツール - Streamlit Web App

## 収録ツール

| ディレクトリ | 内容 |
|---|---|
| `app.py` | AI推定日次決算ツール（収入先・支払先マスターから日次PLを推定する Streamlit アプリ） |
| `slot/` | スロット立ち回り支援ツール（設定判別・天井期待値・差枚シミュレーション・収支管理）。詳細は [slot/README.md](slot/README.md) |

```bash
pip install -r requirements.txt

streamlit run app.py        # 日次決算ツール
streamlit run slot/app.py   # スロット立ち回り支援ツール
```
