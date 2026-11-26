"""期待値まわりの計算。

- 交換率を踏まえた時給換算
- 機種モデルからのモンテカルロ（差枚分布・勝率・資金ショート確率）
- 天井狙いの期待値と「何Gから打てるか」の損益分岐点

標準ライブラリだけで動く。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_TRIALS = 20_000


@dataclass(frozen=True)
class Exchange:
    """貸出単価と換金単価。既定は等価（20円 / 20円）。"""

    rental_yen: float = 20.0
    payout_yen: float = 20.0

    def __post_init__(self) -> None:
        if self.rental_yen <= 0 or self.payout_yen <= 0:
            raise ValueError("単価は正の値にしてください")
        if self.payout_yen > self.rental_yen:
            raise ValueError("換金単価が貸出単価を上回る設定は想定していません")

    @classmethod
    def from_medals_per_100yen(cls, medals_per_100yen: float, rental_yen: float = 20.0) -> "Exchange":
        """``5.6`` 枚交換のような指定から作る（100円で何枚交換できるか）。"""
        if medals_per_100yen <= 0:
            raise ValueError("交換枚数は正の値にしてください")
        return cls(rental_yen=rental_yen, payout_yen=100.0 / medals_per_100yen)

    @property
    def is_even(self) -> bool:
        return math.isclose(self.rental_yen, self.payout_yen)

    def cash(self, diff_medals: float) -> float:
        """差枚を現金に換算する。

        持ちメダル遊技を前提に、プラス分は換金単価、マイナス分は貸出単価で評価する。
        非等価ではこの非対称性があるため、機械割 100% ちょうどでも収支はマイナスになる。
        """
        return diff_medals * (self.payout_yen if diff_medals >= 0 else self.rental_yen)


def hourly_balance(
    payout_rate: float,
    games_per_hour: float,
    medals_per_game: float = 3.0,
    exchange: Exchange | None = None,
) -> dict[str, float]:
    """機械割から 1 時間あたりの期待収支を出す。

    非等価の場合、実際の期待収支は「換金単価評価」と「貸出単価評価」の間に入る。
    正確な値は勝率と分散に依存するので :func:`simulate_diffs` 側で評価する。
    """
    if payout_rate <= 0:
        raise ValueError("機械割は正の値（1.05 または 105 の形式）にしてください")
    if payout_rate > 5.0:  # 105 のような百分率で渡された場合
        payout_rate = payout_rate / 100.0
    exchange = exchange or Exchange()

    inserted = medals_per_game * games_per_hour
    acquired = inserted * payout_rate
    diff = acquired - inserted
    return {
        "投入枚数/時": inserted,
        "獲得枚数/時": acquired,
        "差枚/時": diff,
        "収支/時(換金単価評価)": diff * exchange.payout_yen,
        "収支/時(貸出単価評価)": diff * exchange.rental_yen,
    }


@dataclass(frozen=True)
class SessionModel:
    """1 台を打ち続けたときの差枚をざっくり再現するモデル。

    Attributes
    ----------
    hit_probability:
        通常時 1G あたりの初当り確率（例: 1/300 なら ``1/300``）。
    average_payout:
        初当り 1 回あたりの平均獲得枚数（純増ベース）。
    payout_sd:
        その獲得枚数のばらつき（標準偏差）。0 なら毎回同じ枚数。
    net_loss_per_game:
        通常時 1G あたりの純減枚数。コイン持ちから作る場合は
        :meth:`from_coin_persistence` を使う。
    """

    hit_probability: float
    average_payout: float
    payout_sd: float = 0.0
    net_loss_per_game: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.hit_probability < 1.0:
            raise ValueError("初当り確率は 0 < p < 1 にしてください")
        if self.average_payout <= 0:
            raise ValueError("平均獲得枚数は正の値にしてください")
        if self.payout_sd < 0:
            raise ValueError("標準偏差は 0 以上にしてください")
        if self.net_loss_per_game < 0:
            raise ValueError("純減枚数は 0 以上にしてください")

    @classmethod
    def from_coin_persistence(
        cls,
        games_per_50_medals: float,
        hit_probability: float,
        average_payout: float,
        payout_sd: float = 0.0,
    ) -> "SessionModel":
        """コイン持ち（50枚あたりのゲーム数）から純減を逆算して作る。"""
        if games_per_50_medals <= 0:
            raise ValueError("コイン持ちは正の値にしてください")
        return cls(
            hit_probability=hit_probability,
            average_payout=average_payout,
            payout_sd=payout_sd,
            net_loss_per_game=50.0 / games_per_50_medals,
        )

    def expected_diff(self, normal_games: int) -> float:
        """通常時 ``normal_games`` ゲーム消化したときの期待差枚（解析解）。"""
        return (
            normal_games * self.hit_probability * self.average_payout
            - normal_games * self.net_loss_per_game
        )


def _poisson(mean: float, rng: random.Random) -> int:
    """ポアソン乱数。

    初当り確率は 1/100〜1/1000 程度と小さいため、二項分布の近似として使える。
    """
    if mean <= 0:
        return 0
    if mean < 30.0:  # Knuth の方法
        limit = math.exp(-mean)
        count = 0
        product = 1.0
        while True:
            count += 1
            product *= rng.random()
            if product <= limit:
                return count - 1
    return max(0, round(rng.gauss(mean, math.sqrt(mean))))


def simulate_diffs(
    model: SessionModel,
    normal_games: int,
    trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
) -> list[float]:
    """通常時 ``normal_games`` ゲーム打ったときの差枚を ``trials`` 回サンプリングする。"""
    if normal_games <= 0:
        raise ValueError("ゲーム数は 1 以上にしてください")
    if trials <= 0:
        raise ValueError("試行回数は 1 以上にしてください")

    rng = random.Random(seed)
    mean_hits = model.hit_probability * normal_games
    cost = model.net_loss_per_game * normal_games

    use_gamma = model.payout_sd > 0
    if use_gamma:  # 平均と標準偏差からガンマ分布の形状・尺度を決める
        shape = (model.average_payout / model.payout_sd) ** 2
        scale = model.payout_sd**2 / model.average_payout

    diffs: list[float] = []
    for _ in range(trials):
        hits = _poisson(mean_hits, rng)
        if hits == 0:
            gain = 0.0
        elif use_gamma:
            # 独立なガンマ変数の和は形状パラメータの和になる
            gain = rng.gammavariate(shape * hits, scale)
        else:
            gain = hits * model.average_payout
        diffs.append(gain - cost)
    return diffs


def quantile(values: Sequence[float], q: float) -> float:
    """線形補間の分位点。``values`` はソート済みでなくてよい。"""
    if not values:
        return math.nan
    if not 0.0 <= q <= 1.0:
        raise ValueError("q は 0〜1 にしてください")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def session_stats(
    diffs: Sequence[float],
    exchange: Exchange | None = None,
    bankroll_yen: float | None = None,
) -> dict[str, float]:
    """差枚サンプルから、収支・勝率・分位点をまとめる。"""
    if not diffs:
        raise ValueError("サンプルが空です")
    exchange = exchange or Exchange()

    size = len(diffs)
    mean_diff = sum(diffs) / size
    variance = sum((value - mean_diff) ** 2 for value in diffs) / size
    cash_values = [exchange.cash(value) for value in diffs]
    mean_cash = sum(cash_values) / size
    naive_cash = mean_diff * exchange.payout_yen

    stats = {
        "試行回数": float(size),
        "期待差枚": mean_diff,
        "差枚の標準偏差": math.sqrt(variance),
        "勝率": sum(1 for value in diffs if value > 0) / size,
        "期待収支(円)": mean_cash,
        "参考:差枚×換金単価(円)": naive_cash,
        "交換率による目減り(円)": mean_cash - naive_cash,
        "下位5%差枚": quantile(diffs, 0.05),
        "下位25%差枚": quantile(diffs, 0.25),
        "中央値差枚": quantile(diffs, 0.50),
        "上位75%差枚": quantile(diffs, 0.75),
        "上位95%差枚": quantile(diffs, 0.95),
    }

    if bankroll_yen is not None:
        if bankroll_yen <= 0:
            raise ValueError("資金は正の値にしてください")
        # 終了時点で必要になる現金が資金を超えた割合。
        # 途中経過ではもっと沈むことがあるため、実際の破産確率の下限として読む。
        shortfalls = sum(
            1 for value in diffs if value < 0 and -value * exchange.rental_yen > bankroll_yen
        )
        stats["資金ショート確率"] = shortfalls / size
    return stats


@dataclass(frozen=True)
class CeilingModel:
    """天井狙いの簡易モデル。

    通常時のハザード（1G あたりの当選確率）を一定とみなす近似。実機はモードや
    ゾーンで当選率が変動するため、ゾーンの濃い機種ほど誤差が出る点に注意。
    """

    ceiling_games: int
    hit_probability: float
    average_payout: float
    ceiling_payout: float
    net_loss_per_game: float = 2.0

    def __post_init__(self) -> None:
        if self.ceiling_games <= 0:
            raise ValueError("天井ゲーム数は 1 以上にしてください")
        if not 0.0 < self.hit_probability < 1.0:
            raise ValueError("当選確率は 0 < p < 1 にしてください")
        if self.average_payout < 0 or self.ceiling_payout < 0:
            raise ValueError("獲得枚数は 0 以上にしてください")
        if self.net_loss_per_game < 0:
            raise ValueError("純減枚数は 0 以上にしてください")


def ceiling_ev(
    model: CeilingModel,
    current_games: int,
    exchange: Exchange | None = None,
) -> dict[str, float]:
    """現在 ``current_games`` から打ち始めたときの期待値。

    当選（自力 or 天井）したら獲得して終了、という前提で計算する。
    """
    if current_games < 0:
        raise ValueError("現在ゲーム数は 0 以上にしてください")
    exchange = exchange or Exchange()

    remaining = max(0, model.ceiling_games - current_games)
    miss_probability = (1.0 - model.hit_probability) ** remaining
    self_hit_probability = 1.0 - miss_probability
    # E[min(幾何分布, remaining)] = (1 - (1-p)^remaining) / p
    expected_games = self_hit_probability / model.hit_probability
    invested = expected_games * model.net_loss_per_game
    expected_gain = (
        self_hit_probability * model.average_payout + miss_probability * model.ceiling_payout
    )
    diff = expected_gain - invested

    return {
        "残りゲーム数": float(remaining),
        "自力当選率": self_hit_probability,
        "天井到達率": miss_probability,
        "期待消化ゲーム数": expected_games,
        "期待投資枚数": invested,
        "期待獲得枚数": expected_gain,
        "期待差枚": diff,
        "期待収支(円)": diff * exchange.payout_yen,
    }


def breakeven_start_games(model: CeilingModel, step: int = 1) -> int | None:
    """期待差枚がプラスに転じる最小のゲーム数（＝狙い目ライン）。

    見つからなければ ``None``（その条件では天井直前でも期待値がプラスにならない）。
    """
    if step <= 0:
        raise ValueError("step は 1 以上にしてください")
    for games in range(0, model.ceiling_games, step):
        if ceiling_ev(model, games)["期待差枚"] >= 0:
            return games
    return None

def hourly_from_payout_rate(
    payout_rate: float,
    games_per_hour: float = 600.0,
    medals_per_game: float = 3.0,
    exchange: Exchange | None = None,
) -> float:
    """機械割から時給（円）を出す。

    等価・600G/h なら ``36,000 × (機械割 − 1)``。機械割の小数点以下を 36,000 倍すれば
    時給になる、という換算がそのまま使える。
    """
    if payout_rate > 5.0:  # 105 のような百分率で渡された場合
        payout_rate = payout_rate / 100.0
    if payout_rate <= 0:
        raise ValueError("機械割は正の値にしてください")
    exchange = exchange or Exchange()
    diff_per_hour = medals_per_game * games_per_hour * (payout_rate - 1.0)
    return diff_per_hour * exchange.payout_yen


def breakeven_payout_rate(
    exchange: Exchange,
    loss_medals_per_game: float,
    medals_per_game: float = 3.0,
) -> float:
    """現金投資でこなす区間の、損益分岐となる機械割。

    メダルは機械内で循環するので、現金で買うのは負けた分だけ。よって

        EV = 換金単価 × E[差枚] − (貸出単価 − 換金単価) × E[負け枚数]

    これを 0 にする機械割を返す。``loss_medals_per_game`` はその区間の
    1 ゲームあたり期待負け枚数（通常時の純減 × 非当選率が目安）。

    等価、または貯メダル・持ちメダルでこなす場合は単価差が無いので 1.0 を返す。
    """
    if loss_medals_per_game < 0:
        raise ValueError("負け枚数は 0 以上にしてください")
    gap = exchange.rental_yen - exchange.payout_yen
    if gap <= 0:
        return 1.0
    return 1.0 + (gap / exchange.payout_yen) * loss_medals_per_game / medals_per_game


def procedure_value(
    payout_rate: float,
    games: float,
    exchange: Exchange | None = None,
    medals_per_game: float = 3.0,
    games_per_hour: float = 600.0,
    overhead_minutes: float = 0.0,
    loss_medals_per_game: float | None = None,
) -> dict[str, float]:
    """「機械割○%の手順」を 1 台あたりの実額と実効時給に変換する。

    立ち回り情報で語られる機械割は区間の比率なので、消化ゲーム数が短いと
    率が高くても実額は小さい。台探し・移動・判別にかかる ``overhead_minutes``
    を入れた実効時給まで出して、その差を見えるようにする。
    """
    if payout_rate > 5.0:
        payout_rate = payout_rate / 100.0
    if games <= 0:
        raise ValueError("消化ゲーム数は 1 以上にしてください")
    if overhead_minutes < 0:
        raise ValueError("付帯時間は 0 以上にしてください")
    exchange = exchange or Exchange()

    inserted = medals_per_game * games
    diff = inserted * (payout_rate - 1.0)
    play_minutes = games / games_per_hour * 60.0
    total_minutes = play_minutes + overhead_minutes

    result = {
        "機械割": payout_rate,
        "投入枚数": inserted,
        "期待差枚": diff,
        "期待収支(円)": diff * exchange.payout_yen,
        "消化時間(分)": play_minutes,
        "拘束時間(分)": total_minutes,
        "名目時給(円)": hourly_from_payout_rate(
            payout_rate, games_per_hour, medals_per_game, exchange
        ),
        "実効時給(円)": diff * exchange.payout_yen * 60.0 / total_minutes,
    }
    if loss_medals_per_game is not None:
        breakeven = breakeven_payout_rate(exchange, loss_medals_per_game, medals_per_game)
        result["現金投資の損益分岐機械割"] = breakeven
        result["損益分岐との差"] = payout_rate - breakeven
    return result

def zone_hazard(
    base_probability: float,
    heaven_games: int = 0,
    heaven_multiplier: float = 1.0,
    zone_every: int = 0,
    zone_multiplier: float = 1.0,
) -> Callable[[int], float]:
    """実機によくある「天国＋規定ゲーム数ゾーン」形のハザード関数を組む。

    ``heaven_games`` ゲームまでは ``heaven_multiplier`` 倍、``zone_every`` の倍数の
    ゲームでは ``zone_multiplier`` 倍、それ以外は ``base_probability`` そのまま。

    全体の当選率を実機に合わせたい場合は :func:`calibrate_hazard` を通す。
    """
    if not 0.0 < base_probability < 1.0:
        raise ValueError("基準確率は 0 < p < 1 にしてください")
    if heaven_games < 0 or zone_every < 0:
        raise ValueError("ゲーム数は 0 以上にしてください")
    if heaven_multiplier <= 0 or zone_multiplier <= 0:
        raise ValueError("倍率は正の値にしてください")

    def hazard(game: int) -> float:
        probability = base_probability
        if heaven_games and game <= heaven_games:
            probability *= heaven_multiplier
        elif zone_every and game % zone_every == 0:
            probability *= zone_multiplier
        return min(probability, 1.0)

    return hazard


def _hazard_value(hazard: Callable[[int], float] | Sequence[float], game: int) -> float:
    """ハザードを関数でも配列でも受け取れるようにする。"""
    if callable(hazard):
        value = hazard(game)
    else:
        index = game - 1
        value = hazard[index] if 0 <= index < len(hazard) else 0.0
    return min(max(float(value), 0.0), 1.0)


def total_hit_probability(
    hazard: Callable[[int], float] | Sequence[float],
    ceiling_games: int,
    current_games: int = 0,
) -> float:
    """``current_games`` から天井までに自力当選する確率。"""
    survive = 1.0
    for game in range(current_games + 1, ceiling_games + 1):
        survive *= 1.0 - _hazard_value(hazard, game)
    return 1.0 - survive


def calibrate_hazard(
    builder: Callable[[float], Callable[[int], float]],
    target_hit_probability: float,
    ceiling_games: int,
    iterations: int = 80,
) -> Callable[[int], float]:
    """自力当選率が実機の値に一致するよう、ハザード全体のスケールを合わせる。

    ``builder`` は基準確率を受け取ってハザード関数を返す呼び出し可能オブジェクト。
    ゾーンの形は保ったまま、全体の当選率だけを合わせられる。
    """
    if not 0.0 < target_hit_probability < 1.0:
        raise ValueError("目標当選率は 0 < p < 1 にしてください")
    low, high = 1e-9, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if total_hit_probability(builder(middle), ceiling_games) < target_hit_probability:
            low = middle
        else:
            high = middle
    return builder((low + high) / 2)


def hazard_ceiling_ev(
    hazard: Callable[[int], float] | Sequence[float],
    ceiling_games: int,
    current_games: int = 0,
    average_payout: float = 0.0,
    ceiling_payout: float = 0.0,
    net_loss_per_game: float = 2.0,
    exchange: Exchange | None = None,
) -> dict[str, float]:
    """ゲームごとの当選確率を与えて、天井狙いの期待値を厳密に計算する。

    :func:`ceiling_ev` はハザード一定を仮定した閉じた式だが、天国やゾーンのある機種では
    その近似が大きくずれる。当選が前方に寄った機種では浅いゲーム数の期待値を過小評価し、
    ゾーンを通過した直後では逆に過大評価する（ズレの符号が変わる）。
    ゾーンが効く機種ではこちらを使う。
    """
    if ceiling_games <= 0:
        raise ValueError("天井ゲーム数は 1 以上にしてください")
    if current_games < 0 or current_games > ceiling_games:
        raise ValueError("現在ゲーム数は 0 以上、天井以下にしてください")
    exchange = exchange or Exchange()

    survive = 1.0
    expected_games = 0.0
    self_hit = 0.0
    for game in range(current_games + 1, ceiling_games + 1):
        probability = _hazard_value(hazard, game)
        expected_games += survive * probability * (game - current_games)
        self_hit += survive * probability
        survive *= 1.0 - probability
    expected_games += survive * (ceiling_games - current_games)

    invested = expected_games * net_loss_per_game
    expected_gain = self_hit * average_payout + survive * ceiling_payout
    diff = expected_gain - invested

    return {
        "残りゲーム数": float(ceiling_games - current_games),
        "自力当選率": self_hit,
        "天井到達率": survive,
        "期待消化ゲーム数": expected_games,
        "期待投資枚数": invested,
        "期待獲得枚数": expected_gain,
        "期待差枚": diff,
        "期待収支(円)": diff * exchange.payout_yen,
    }


def hazard_breakeven_start_games(
    hazard: Callable[[int], float] | Sequence[float],
    ceiling_games: int,
    average_payout: float,
    ceiling_payout: float,
    net_loss_per_game: float = 2.0,
    step: int = 1,
) -> int | None:
    """ゾーンを考慮した狙い目ライン。

    ゾーンのある機種では期待値がゲーム数に対して単調に増えないので、
    最初にプラスへ転じる点ではなく、**そこから天井まで一度もマイナスに戻らない**
    最小のゲーム数を返す。見つからなければ ``None``。
    """
    if step <= 0:
        raise ValueError("step は 1 以上にしてください")
    candidates = list(range(0, ceiling_games, step))
    answer: int | None = None
    for games in reversed(candidates):
        value = hazard_ceiling_ev(
            hazard, ceiling_games, games,
            average_payout=average_payout, ceiling_payout=ceiling_payout,
            net_loss_per_game=net_loss_per_game,
        )["期待差枚"]
        if value >= 0:
            answer = games
        else:
            break
    return answer
