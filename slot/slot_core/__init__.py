"""スロット立ち回り支援のコアロジック。

標準ライブラリだけで動くので、Streamlit が無い環境（CI・ノートブック・CLI）
からもそのまま使える。表示層は ``slot/app.py`` 側に閉じ込めている。
"""

from . import bayes, ev, hall, ledger, stats

__all__ = ["bayes", "ev", "hall", "ledger", "stats"]
