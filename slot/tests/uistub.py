"""``slot/app.py`` を本物の streamlit / pandas なしで実行するためのスタブ。

本物の代替品ではなく、**本物が受け付けない書き方を見つける検査器**として作ってある。
PyPI に到達できない環境でも表示層を実際に動かせるようにするのが目的。

検出できるもの:

- ``number_input`` / ``slider`` の数値型混在（Streamlit の MixedNumericTypes 相当）
- ``number_input`` / ``slider`` の ``value`` が ``min_value`` / ``max_value`` の外に出る指定
- 同一ページ内でのウィジェットキー重複（DuplicateWidgetID 相当）
- ``DataFrame`` の存在しない列へのアクセス・整形指定（KeyError 相当）
- Styler の書式指定と実際の値の型が噛み合わないこと（例: 文字列に ``{:+,.0f}``）
- グラフへ数値でない列を渡すこと

検出できないもの（本物でしか分からない範囲）:

- ウィジェットの見た目・レイアウト・再実行時のセッション状態の挙動
- ``read_csv`` の型推論の細部（このスタブは数値に見える文字列だけ数値化する）
"""

from __future__ import annotations

import contextlib
import csv
import io
import math
import sys
import types

MISSING = object()


# --- pandas ----------------------------------------------------------------
def _is_missing(value: object) -> bool:
    if value is None or value is MISSING:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _maybe_number(text: str) -> object:
    stripped = text.strip()
    if stripped == "":
        return ""
    try:
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
        return float(stripped)
    except ValueError:
        return text


class Series:
    """1 列だけの最小実装。グラフと反復に使う。"""

    def __init__(self, values: list, index: list, name: str | None = None) -> None:
        self.values = list(values)
        self.index = list(index)
        self.name = name

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def tolist(self) -> list:
        return list(self.values)


class Styler:
    """``DataFrame.style`` の最小実装。書式は描画時に実際に適用する。"""

    def __init__(self, frame: "DataFrame") -> None:
        self.frame = frame
        self._formats: list[tuple[object, object]] = []

    def format(self, formatter=None, subset=None, **_kwargs) -> "Styler":
        if isinstance(formatter, dict):
            unknown = [key for key in formatter if key not in self.frame.columns]
            if unknown:
                raise KeyError(f"Styler.format: 存在しない列 {unknown}（実際の列: {self.frame.columns}）")
        if subset is not None:
            names = [subset] if isinstance(subset, str) else list(subset)
            unknown = [name for name in names if name not in self.frame.columns]
            if unknown:
                raise KeyError(f"Styler.format(subset=): 存在しない列 {unknown}（実際の列: {self.frame.columns}）")
        self._formats.append((formatter, subset))
        return self

    def render_values(self) -> list[str]:
        """書式を実際に適用する。型が噛み合わなければここで例外になる。"""
        rendered: list[str] = []
        for formatter, subset in self._formats:
            if isinstance(formatter, dict):
                pairs = [(column, spec) for column, spec in formatter.items()]
            elif formatter is None:
                continue
            else:
                names = (
                    self.frame.columns
                    if subset is None
                    else ([subset] if isinstance(subset, str) else list(subset))
                )
                pairs = [(column, formatter) for column in names]
            for column, spec in pairs:
                for value in self.frame[column]:
                    if _is_missing(value):
                        continue
                    if callable(spec):
                        rendered.append(str(spec(value)))
                    else:
                        try:
                            rendered.append(spec.format(value))
                        except (ValueError, TypeError) as error:
                            raise TypeError(
                                f"列『{column}』の書式 {spec!r} に値 {value!r} "
                                f"({type(value).__name__}) を渡せません: {error}"
                            ) from error
        return rendered


class DataFrame:
    def __init__(self, data=None, index=None) -> None:
        self._columns: list[str] = []
        self._data: dict[str, list] = {}
        self.index: list = []
        self.index_name: str | None = None

        if data is None:
            data = {}
        if isinstance(data, DataFrame):
            self._columns = list(data._columns)
            self._data = {key: list(value) for key, value in data._data.items()}
            self.index = list(data.index)
            self.index_name = data.index_name
            return

        if isinstance(data, dict):
            # dict の値が dict なら「列 -> (行ラベル -> 値)」
            if data and all(isinstance(value, dict) for value in data.values()):
                labels: list = []
                for mapping in data.values():
                    for label in mapping:
                        if label not in labels:
                            labels.append(label)
                self.index = labels
                for column, mapping in data.items():
                    self._columns.append(column)
                    self._data[column] = [mapping.get(label, math.nan) for label in labels]
            else:
                lengths = {len(list(value)) for value in data.values()} or {0}
                if len(lengths) > 1:
                    raise ValueError(f"列の長さが揃っていません: {lengths}")
                size = lengths.pop()
                for column, values in data.items():
                    self._columns.append(column)
                    self._data[column] = list(values)
                self.index = list(index) if index is not None else list(range(size))
        elif isinstance(data, list):
            for record in data:
                if not isinstance(record, dict):
                    raise TypeError("DataFrame(list) は dict のリストのみ対応します")
                for column in record:
                    if column not in self._columns:
                        self._columns.append(column)
                        self._data[column] = []
            for position, record in enumerate(data):
                for column in self._columns:
                    self._data[column].append(record.get(column, math.nan))
                del position
            self.index = list(index) if index is not None else list(range(len(data)))
        else:
            raise TypeError(f"未対応の DataFrame 入力: {type(data).__name__}")

        if index is not None and len(self.index) != len(list(index)):
            raise ValueError("index の長さが行数と一致しません")

    # -- 基本情報
    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def empty(self) -> bool:
        return not self._columns or not self.index

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, column: object) -> bool:
        return column in self._columns

    def __getitem__(self, column):
        if isinstance(column, list):
            result = DataFrame({name: self[name].values for name in column})
            result.index = list(self.index)
            result.index_name = self.index_name
            return result
        if column not in self._data:
            raise KeyError(f"列『{column}』がありません（実際の列: {self._columns}）")
        return Series(self._data[column], self.index, name=column)

    def __setitem__(self, column: str, values) -> None:
        values = list(values.values) if isinstance(values, Series) else list(values)
        if self._columns and len(values) != len(self.index):
            raise ValueError(f"列『{column}』の長さ {len(values)} が行数 {len(self.index)} と違います")
        if column not in self._data:
            self._columns.append(column)
        self._data[column] = values

    # -- 変形
    def set_index(self, column: str) -> "DataFrame":
        if column not in self._data:
            raise KeyError(f"set_index: 列『{column}』がありません（実際の列: {self._columns}）")
        result = DataFrame(self)
        result.index = list(self._data[column])
        result.index_name = column
        result._columns = [name for name in self._columns if name != column]
        result._data = {name: list(self._data[name]) for name in result._columns}
        return result

    def rename(self, columns: dict | None = None) -> "DataFrame":
        result = DataFrame(self)
        if not columns:
            return result
        result._columns = [columns.get(name, name) for name in self._columns]
        result._data = {columns.get(name, name): list(values) for name, values in self._data.items()}
        return result

    def sort_values(self, column: str) -> "DataFrame":
        if column not in self._data:
            raise KeyError(f"sort_values: 列『{column}』がありません（実際の列: {self._columns}）")
        order = sorted(range(len(self.index)), key=lambda position: self._data[column][position])
        result = DataFrame(self)
        result.index = [self.index[position] for position in order]
        result._data = {
            name: [values[position] for position in order] for name, values in self._data.items()
        }
        return result

    def join(self, other: "DataFrame") -> "DataFrame":
        overlap = [name for name in other.columns if name in self._columns]
        if overlap:
            raise ValueError(f"join: 列名が重複しています {overlap}")
        result = DataFrame(self)
        position_of = {label: position for position, label in enumerate(other.index)}
        for name in other.columns:
            source = other._data[name]
            result._columns.append(name)
            result._data[name] = [
                source[position_of[label]] if label in position_of else math.nan
                for label in self.index
            ]
        return result

    def dropna(self, subset: list[str] | None = None) -> "DataFrame":
        names = list(subset) if subset else list(self._columns)
        unknown = [name for name in names if name not in self._data]
        if unknown:
            raise KeyError(f"dropna: 列 {unknown} がありません（実際の列: {self._columns}）")
        keep = [
            position
            for position in range(len(self.index))
            if not any(_is_missing(self._data[name][position]) for name in names)
        ]
        result = DataFrame(self)
        result.index = [self.index[position] for position in keep]
        result._data = {
            name: [values[position] for position in keep] for name, values in self._data.items()
        }
        return result

    # -- 出力
    def to_dict(self, orient: str = "records") -> list[dict]:
        if orient != "records":
            raise NotImplementedError(f"to_dict(orient={orient!r}) は未対応")
        return [
            {name: self._data[name][position] for name in self._columns}
            for position in range(len(self.index))
        ]

    def to_csv(self, index: bool = True) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        header = ([self.index_name or ""] if index else []) + self._columns
        writer.writerow(header)
        for position in range(len(self.index)):
            row = ([self.index[position]] if index else []) + [
                self._data[name][position] for name in self._columns
            ]
            writer.writerow(row)
        return buffer.getvalue()

    @property
    def style(self) -> Styler:
        return Styler(self)


def read_csv(source) -> DataFrame:
    if hasattr(source, "getvalue"):
        raw = source.getvalue()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    elif hasattr(source, "read"):
        text = source.read()
    else:
        with open(source, encoding="utf-8-sig") as handle:
            text = handle.read()
    reader = csv.DictReader(io.StringIO(text))
    rows = [
        {key: _maybe_number(value) for key, value in record.items() if key is not None}
        for record in reader
    ]
    return DataFrame(rows)


def build_pandas_module() -> types.ModuleType:
    module = types.ModuleType("pandas")
    module.DataFrame = DataFrame
    module.Series = Series
    module.read_csv = read_csv
    module.isna = _is_missing
    module.__version__ = "0.0.0-stub"
    return module


# --- streamlit --------------------------------------------------------------
class WidgetError(AssertionError):
    """本物の Streamlit がエラーにする指定を見つけたときに送出する。"""


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value) -> None:
        self[name] = value


class UploadedFile:
    """``st.file_uploader`` の戻り値を模したもの。"""

    def __init__(self, path: str, name: str | None = None) -> None:
        with open(path, "rb") as handle:
            self._raw = handle.read()
        self.name = name or path.rsplit("/", 1)[-1]

    def getvalue(self) -> bytes:
        return self._raw

    def read(self) -> bytes:
        return self._raw


def _numeric_kind(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return None


class Streamlit:
    """``import streamlit as st`` の置き換え。呼び出しを記録しつつ検査する。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.rendered: list[str] = []
        self.session_state = SessionState()
        self.widget_mode = "default"
        self.uploads: dict[str, UploadedFile] = {}
        self._keys: dict[str, str] = {}
        self._auto: set[tuple[str, str]] = set()
        self.sidebar = self

    # -- テスト側の操作
    def reset(self, widget_mode: str = "default", uploads: dict | None = None) -> None:
        self.calls.clear()
        self.rendered.clear()
        self.widget_mode = widget_mode
        self.uploads = dict(uploads or {})
        self._keys.clear()
        self._auto.clear()

    def names_called(self) -> set[str]:
        return {name for name, _args, _kwargs in self.calls}

    def texts(self) -> list[str]:
        return [
            args[0]
            for name, args, _kwargs in self.calls
            if args and isinstance(args[0], str) and name not in {"metric"}
        ]

    # -- 内部
    def _record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def _register(self, widget: str, label: object, key: object) -> None:
        if not isinstance(label, str):
            raise WidgetError(f"{widget}: ラベルが文字列ではありません ({label!r})")
        if key is not None:
            if key in self._keys:
                raise WidgetError(f"ウィジェットキー {key!r} が同一ページ内で重複しています")
            self._keys[str(key)] = label
        else:
            signature = (widget, label)
            if signature in self._auto:
                raise WidgetError(
                    f"{widget}『{label}』がキー無しで重複しています（DuplicateWidgetID になります）"
                )
            self._auto.add(signature)

    def _pick_number(self, widget, label, min_value, max_value, value, step, key):
        self._register(widget, label, key)
        kinds = {
            kind
            for kind in (
                _numeric_kind(min_value),
                _numeric_kind(max_value),
                _numeric_kind(value),
                _numeric_kind(step),
            )
            if kind is not None
        }
        if len(kinds) > 1:
            raise WidgetError(
                f"{widget}『{label}』: int と float が混在しています "
                f"(min={min_value!r}, max={max_value!r}, value={value!r}, step={step!r})"
            )
        if value is not None:
            if min_value is not None and value < min_value:
                raise WidgetError(f"{widget}『{label}』: value {value} が min_value {min_value} 未満です")
            if max_value is not None and value > max_value:
                raise WidgetError(f"{widget}『{label}』: value {value} が max_value {max_value} を超えています")
        self._record(widget, label)
        if self.widget_mode == "min" and min_value is not None:
            return min_value
        if self.widget_mode == "max" and max_value is not None:
            return max_value
        return value

    def _pick_option(self, widget, label, options, key):
        self._register(widget, label, key)
        options = list(options)
        if not options:
            raise WidgetError(f"{widget}『{label}』: 選択肢が空です")
        self._record(widget, label, options=options)
        return options[-1] if self.widget_mode == "max" else options[0]

    def _check_chart(self, name: str, data) -> None:
        if isinstance(data, Styler):
            raise WidgetError(f"{name}: Styler は渡せません")
        if isinstance(data, Series):
            columns = {data.name or "値": data.values}
        elif isinstance(data, DataFrame):
            if data.empty:
                raise WidgetError(f"{name}: 空の DataFrame は描画できません")
            columns = {name_: data[name_].values for name_ in data.columns}
        else:
            raise WidgetError(f"{name}: DataFrame か Series を渡してください ({type(data).__name__})")
        for column, values in columns.items():
            for value in values:
                if _is_missing(value):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise WidgetError(
                        f"{name}: 列『{column}』に数値でない値 {value!r} "
                        f"({type(value).__name__}) があります"
                    )

    # -- レイアウト
    def set_page_config(self, **kwargs):
        self._record("set_page_config", **kwargs)

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(list(spec))
        if count < 1:
            raise WidgetError(f"st.columns: 列数が {count} です")
        self._record("columns", count)
        return [self for _ in range(count)]

    def tabs(self, labels):
        labels = list(labels)
        if not labels:
            raise WidgetError("st.tabs: ラベルが空です")
        self._record("tabs", labels)
        return [self for _ in labels]

    @contextlib.contextmanager
    def expander(self, label, **kwargs):
        self._record("expander", label)
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    # -- テキスト
    def _text(self, name):
        def handler(body="", **kwargs):
            if not isinstance(body, str):
                raise WidgetError(f"st.{name}: 文字列以外が渡されました ({type(body).__name__})")
            self.rendered.append(body)
            self._record(name, body, **kwargs)

        return handler

    def __getattr__(self, name):
        if name in {
            "title", "header", "subheader", "caption", "write",
            "info", "warning", "error", "success", "markdown", "text",
        }:
            return self._text(name)
        raise AttributeError(f"スタブに st.{name} がありません（app.py が新しい API を使っています）")

    # -- 表示
    def metric(self, label, value=None, delta=None, **kwargs):
        if not isinstance(label, str):
            raise WidgetError(f"st.metric: ラベルが文字列ではありません ({label!r})")
        if value is not None and not isinstance(value, (str, int, float)):
            raise WidgetError(f"st.metric『{label}』: 値が文字列・数値ではありません ({type(value).__name__})")
        self._record("metric", label, value, delta)

    def dataframe(self, data, **kwargs):
        if isinstance(data, Styler):
            self.rendered.extend(data.render_values())
        elif not isinstance(data, (DataFrame, Series)):
            raise WidgetError(f"st.dataframe: DataFrame/Styler を渡してください ({type(data).__name__})")
        self._record("dataframe", type(data).__name__)

    def data_editor(self, data, **kwargs):
        self._register("data_editor", "data_editor", kwargs.get("key"))
        if not isinstance(data, DataFrame):
            raise WidgetError(f"st.data_editor: DataFrame を渡してください ({type(data).__name__})")
        self._record("data_editor", len(data))
        return data

    def line_chart(self, data, **kwargs):
        self._check_chart("line_chart", data)
        self._record("line_chart")

    def bar_chart(self, data, **kwargs):
        self._check_chart("bar_chart", data)
        self._record("bar_chart")

    def area_chart(self, data, **kwargs):
        self._check_chart("area_chart", data)
        self._record("area_chart")

    def download_button(self, label, data, file_name=None, mime=None, **kwargs):
        self._register("download_button", label, kwargs.get("key"))
        if not isinstance(data, (bytes, bytearray, str)):
            raise WidgetError(f"st.download_button『{label}』: bytes か str を渡してください ({type(data).__name__})")
        self._record("download_button", label)

    # -- 入力
    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kwargs):
        return self._pick_number(
            "number_input", label, min_value, max_value, value, step, kwargs.get("key")
        )

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **kwargs):
        return self._pick_number("slider", label, min_value, max_value, value, step, kwargs.get("key"))

    def radio(self, label, options, **kwargs):
        return self._pick_option("radio", label, options, kwargs.get("key"))

    def selectbox(self, label, options, **kwargs):
        return self._pick_option("selectbox", label, options, kwargs.get("key"))

    def multiselect(self, label, options, default=None, **kwargs):
        self._register("multiselect", label, kwargs.get("key"))
        options = list(options)
        self._record("multiselect", label)
        if self.widget_mode == "min":
            return []
        if default is None:
            return list(options)
        return list(default) if isinstance(default, (list, tuple)) else [default]

    def checkbox(self, label, value=False, **kwargs):
        self._register("checkbox", label, kwargs.get("key"))
        self._record("checkbox", label)
        if self.widget_mode == "zones":
            return True
        if self.widget_mode == "min":
            return False
        return bool(value)

    def file_uploader(self, label, type=None, **kwargs):
        key = kwargs.get("key")
        self._register("file_uploader", label, key)
        self._record("file_uploader", label)
        return self.uploads.get(str(key))


def build_streamlit_module() -> Streamlit:
    return Streamlit()


@contextlib.contextmanager
def installed():
    """``streamlit`` と ``pandas`` をスタブに差し替える。終了時に必ず元へ戻す。"""
    saved = {name: sys.modules.get(name) for name in ("streamlit", "pandas")}
    fake_st = build_streamlit_module()
    sys.modules["streamlit"] = fake_st
    sys.modules["pandas"] = build_pandas_module()
    try:
        yield fake_st
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
