"""``slot/app.py`` の表示層を実際に実行して検証する。

streamlit / pandas は ``uistub`` のスタブに差し替える。見た目は検証できないが、
**ウィジェット指定・列名・書式・グラフに渡す型**といった「本物なら例外になる」範囲は
これで押さえられる。6 ページすべてを既定値 / 下限 / 上限 / CSV アップロードで走らせる。
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uistub  # noqa: E402

SLOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(SLOT_DIR, "app.py")
DATA_DIR = os.path.join(SLOT_DIR, "data")


def _load_app(module_name: str = "slot_app_under_test"):
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AppPageTest(unittest.TestCase):
    """スタブ上で app.py を import し、各ページを呼び出す。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stack = contextlib.ExitStack()
        cls.st = cls._stack.enter_context(uistub.installed())
        cls.app = _load_app()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stack.close()

    def setUp(self) -> None:
        self.st.reset()

    # -- 前提: スタブが検査器として機能していること ------------------------
    def test_stub_rejects_mixed_numeric_types(self) -> None:
        """int と float の混在を検出できること（Streamlit が実際にエラーにする指定）。"""
        with self.assertRaises(uistub.WidgetError):
            self.st.number_input("混在", min_value=0, max_value=10.0, value=5)

    def test_stub_rejects_value_outside_range(self) -> None:
        with self.assertRaises(uistub.WidgetError):
            self.st.number_input("範囲外", min_value=0, max_value=10, value=700)

    def test_stub_rejects_duplicate_widget_key(self) -> None:
        self.st.number_input("A", min_value=0, max_value=10, value=1, key="dup")
        with self.assertRaises(uistub.WidgetError):
            self.st.number_input("B", min_value=0, max_value=10, value=1, key="dup")

    def test_stub_rejects_unknown_column_in_format(self) -> None:
        frame = sys.modules["pandas"].DataFrame({"あり": [1.0]})
        with self.assertRaises(KeyError):
            frame.style.format({"なし": "{:.1f}"}).render_values()

    def test_stub_rejects_non_numeric_chart_column(self) -> None:
        frame = sys.modules["pandas"].DataFrame({"文字": ["a", "b"]})
        with self.assertRaises(uistub.WidgetError):
            self.st.line_chart(frame)

    def test_stub_applies_formats_and_catches_type_mismatch(self) -> None:
        frame = sys.modules["pandas"].DataFrame({"値": ["文字列"]})
        with self.assertRaises(TypeError):
            frame.style.format({"値": "{:+,.0f}"}).render_values()

    # -- 本題: 全ページが動くこと ------------------------------------------
    def test_page_registry(self) -> None:
        self.assertEqual(len(self.app.PAGES), 6)
        for name, handler in self.app.PAGES.items():
            self.assertTrue(callable(handler), name)

    def test_every_page_runs_with_defaults(self) -> None:
        for name, handler in self.app.PAGES.items():
            with self.subTest(page=name):
                self.st.reset()
                handler()
                self.assertTrue(self.st.calls, f"{name}: 何も描画されていません")

    def test_every_page_runs_at_widget_bounds(self) -> None:
        """下限・上限に振り切った入力でも落ちないこと。"""
        for mode in ("min", "max"):
            for name, handler in self.app.PAGES.items():
                with self.subTest(page=name, mode=mode):
                    self.st.reset(widget_mode=mode)
                    handler()

    def test_main_dispatches(self) -> None:
        self.st.reset()
        self.app.main()
        self.assertIn("radio", self.st.names_called())

    # -- ページ固有の経路 --------------------------------------------------
    def test_ceiling_page_zone_mode_draws_both_curves(self) -> None:
        self.st.reset(widget_mode="zones")
        self.app.page_ceiling()
        self.assertIn("line_chart", self.st.names_called())
        self.assertTrue(
            any("ハザード一定の式だと" in text for text in self.st.rendered),
            "ゾーン考慮時に一定近似との併記が出ていません",
        )

    def test_ceiling_page_flat_mode_has_no_zone_note(self) -> None:
        self.st.reset()
        self.app.page_ceiling()
        self.assertFalse(any("ハザード一定の式だと" in text for text in self.st.rendered))

    def test_setting_estimation_reaches_verdict_when_counts_present(self) -> None:
        self.st.reset(widget_mode="max")
        self.app.page_setting_estimation()
        names = self.st.names_called()
        self.assertIn("bar_chart", names, "事後確率のグラフまで到達していません")
        self.assertIn("metric", names)

    def test_setting_estimation_waits_for_input_by_default(self) -> None:
        self.st.reset()
        self.app.page_setting_estimation()
        self.assertTrue(any("観測回数" in text for text in self.st.rendered))

    def test_hall_page_renders_all_tabs(self) -> None:
        self.st.reset()
        self.app.page_hall()
        tab_calls = [args for name, args, _ in self.st.calls if name == "tabs"]
        self.assertEqual(len(tab_calls), 1)
        self.assertEqual(len(tab_calls[0][0]), 6)

    def test_ledger_page_offers_download(self) -> None:
        self.st.reset()
        self.app.page_ledger()
        self.assertIn("download_button", self.st.names_called())

    # -- CSV アップロード経路 ----------------------------------------------
    def test_uploaded_csv_paths(self) -> None:
        cases = {
            "machine_csv": ("machines_myjuggler5.csv", self.app.page_setting_estimation),
            "ledger_csv": ("ledger_sample.csv", self.app.page_ledger),
            "hall_csv": ("hall_sample.csv", self.app.page_hall),
        }
        for key, (filename, handler) in cases.items():
            with self.subTest(upload=filename):
                self.st.reset(
                    uploads={key: uistub.UploadedFile(os.path.join(DATA_DIR, filename))}
                )
                handler()
                self.assertFalse(
                    [text for text in self.st.rendered if "読めませんでした" in text or "読み込めませんでした" in text],
                    f"{filename} のアップロード経路で読み込みエラーになりました",
                )

    # -- 回帰: 実際に見つかった不具合 --------------------------------------
    def test_probability_input_never_yields_probability_one(self) -> None:
        """1/N の下限が 1 だと p=1.0 になり core の 0<p<1 を破っていた。"""
        for mode in ("default", "min", "max"):
            with self.subTest(mode=mode):
                self.st.reset(widget_mode=mode)
                value = self.app.probability_input("確率 1/N", 300.0, "regression_probability")
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)

    def test_ceiling_default_current_games_fits_shallow_ceiling(self) -> None:
        """天井が 700G 未満の機種で『現在のゲーム数』の既定値が上限を超えていた。"""
        self.st.reset(widget_mode="min")
        self.app.page_ceiling()
        bounds = [
            kwargs
            for name, args, kwargs in self.st.calls
            if name == "number_input" and args and args[0] == "現在のゲーム数"
        ]
        self.assertTrue(bounds, "『現在のゲーム数』の入力欄が描画されていません")

    def test_impossible_inputs_show_message_instead_of_traceback(self) -> None:
        """core の ValueError が画面のエラー表示になること（素の例外で落ちないこと）。"""
        original = self.app.ev.SessionModel

        def exploding(*args, **kwargs):
            raise ValueError("テスト用の検証エラー")

        self.app.ev.SessionModel = exploding
        try:
            self.st.reset()
            self.app.page_simulation()
        finally:
            self.app.ev.SessionModel = original
        self.assertTrue(
            any("入力値では計算できません" in text for text in self.st.rendered),
            "core の検証エラーが画面に出ていません",
        )

    def test_uploaded_broken_csv_is_reported_not_crashed(self) -> None:
        """壊れた CSV はエラー表示で止まり、例外を投げないこと。"""
        broken = os.path.join(DATA_DIR, "machines_sample.csv")
        upload = uistub.UploadedFile(broken)
        upload._raw = b"\xef\xbb\xbf\xe5\x88\x97,\xe3\x81\xaa\xe3\x81\x97\n1,2\n"
        self.st.reset(uploads={"machine_csv": upload})
        self.app.page_setting_estimation()
        self.assertTrue(
            any("機種テーブルを読めませんでした" in text for text in self.st.rendered),
            "不正な CSV がエラー表示されていません",
        )


if __name__ == "__main__":
    unittest.main()
