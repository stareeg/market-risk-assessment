"""
Функции загрузки данных из первоисточников, разрешённых в задании:
  * MOEX ISS  (moex.ru)   — акции, облигации, индексы, срочный рынок;
  * ЦБ РФ      (cbr.ru)    — кривая бескупонной доходности (КБД) и курсы валют.

Все сетевые обращения идут через явные HTTP-запросы к открытым сервисам, без
авторизации. Это позволяет полностью воспроизвести выборку из тетрадки.

Каждая функция возвращает pandas.DataFrame и не пишет на диск — сохранение
в parquet делает тетрадка (так удобнее обращаться к промежуточным результатам).
"""
from __future__ import annotations

import io
import re
import time
import datetime as dt
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests

ISS = "https://iss.moex.com/iss"
HEADERS = {"User-Agent": "Mozilla/5.0 (market-risk-hw data_loader)"}

# Месячные коды фьючерсов (январь..декабрь) по стандарту бирж.
FUT_MONTH_CODES = "FGHJKMNQUVXZ"


# =========================================================================
#  Низкоуровневые помощники MOEX ISS
# =========================================================================
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _iss_get(session: requests.Session, url: str, params: dict | None = None) -> dict:
    """GET к ISS с JSON-ответом и retry на сетевые сбои."""
    params = dict(params or {})
    params.setdefault("iss.meta", "off")
    last = None
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — простой retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ISS GET failed: {url} :: {last}")


def _block_to_df(payload: dict, block: str) -> pd.DataFrame:
    cols = payload[block]["columns"]
    data = payload[block]["data"]
    return pd.DataFrame(data, columns=cols)


def _iss_history_paged(session, url: str, params: dict, block: str = "history") -> pd.DataFrame:
    """Постранично выкачивает блок history (ISS отдаёт по 100 строк за раз)."""
    frames, start = [], 0
    while True:
        p = dict(params)
        p["start"] = start
        payload = _iss_get(session, url, p)
        df = _block_to_df(payload, block)
        if df.empty:
            break
        frames.append(df)
        start += len(df)
        # cursor может отсутствовать при iss.meta=off — ориентируемся на размер пачки
        if len(df) < 100:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =========================================================================
#  1.b / 1.c  ОБЛИГАЦИИ
# =========================================================================
def resolve_ofz_secid(session, number: str) -> str:
    """По короткому номеру выпуска (например '26221') находит полный SECID ОФЗ."""
    payload = _iss_get(session, f"{ISS}/securities.json",
                       {"q": number, "engine": "stock", "market": "bonds"})
    df = _block_to_df(payload, "securities")
    # Берём государственную ОФЗ с этим номером в SECID.
    mask = df["secid"].str.contains(number) & df["secid"].str.startswith("SU")
    cand = df[mask]
    if cand.empty:
        raise ValueError(f"Не найден SECID для ОФЗ {number}")
    return cand.iloc[0]["secid"]


def get_bond_description(session, secid: str) -> dict:
    """Карточка облигации: имя, ISIN, дата погашения, купон, наличие оферты и т.п."""
    payload = _iss_get(session, f"{ISS}/securities/{secid}.json")
    desc = {row[0]: row[2] for row in payload["description"]["data"]}
    return desc


def get_bond_schedule(session, secid: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Расписание выплат: купоны, амортизация номинала, оферты."""
    payload = _iss_get(session, f"{ISS}/securities/{secid}/bondization.json",
                       {"limit": 100})
    coupons = _block_to_df(payload, "coupons")
    amorts = _block_to_df(payload, "amortizations")
    offers = _block_to_df(payload, "offers")
    return coupons, amorts, offers


def get_bond_history(session, secid: str, start: str, end: str,
                     board: str = "TQOB") -> pd.DataFrame:
    """История торгов облигации (доска TQOB — основной режим ОФЗ)."""
    url = f"{ISS}/history/engines/stock/markets/bonds/boards/{board}/securities/{secid}.json"
    cols = ("TRADEDATE", "SHORTNAME", "SECID", "CLOSE", "LEGALCLOSEPRICE",
            "ACCINT", "WAPRICE", "YIELDCLOSE", "DURATION", "FACEVALUE",
            "COUPONVALUE", "NUMTRADES", "VOLUME", "VALUE")
    df = _iss_history_paged(session, url, {
        "from": start, "till": end,
        "history.columns": ",".join(cols),
    })
    return df


# =========================================================================
#  1.d  АКЦИИ   |   1.e  ИНДЕКСЫ
# =========================================================================
def get_share_history(session, ticker: str, start: str, end: str,
                      board: str = "TQBR") -> pd.DataFrame:
    url = f"{ISS}/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
    cols = ("TRADEDATE", "SECID", "CLOSE", "LEGALCLOSEPRICE", "WAPRICE",
            "OPEN", "HIGH", "LOW", "NUMTRADES", "VOLUME", "VALUE")
    return _iss_history_paged(session, url, {
        "from": start, "till": end,
        "history.columns": ",".join(cols),
    })


def get_index_history(session, ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{ISS}/history/engines/stock/markets/index/securities/{ticker}.json"
    cols = ("TRADEDATE", "SECID", "CLOSE", "OPEN", "HIGH", "LOW", "VALUE")
    return _iss_history_paged(session, url, {
        "from": start, "till": end,
        "history.columns": ",".join(cols),
    })


# =========================================================================
#  1.e  BRENT (фронт-месяц из фьючерсов BR на FORTS)
# =========================================================================
def _forts_contract_codes(assetcode: str, start: str, end: str) -> list[str]:
    """Генерирует возможные коды фьючерсов вида BR<месяц><год> на интервале."""
    y0, y1 = int(start[:4]), int(end[:4])
    codes = []
    for year in range(y0 - 1, y1 + 1):          # с запасом на контракты прошлого года
        ydigit = str(year % 10)
        for m in FUT_MONTH_CODES:
            codes.append(f"{assetcode}{m}{ydigit}")
    return codes


def get_futures_contract_history(session, secid: str, start: str, end: str) -> pd.DataFrame:
    url = f"{ISS}/history/engines/futures/markets/forts/securities/{secid}.json"
    cols = ("TRADEDATE", "SECID", "CLOSE", "SETTLEPRICE", "VOLUME",
            "OPENPOSITION", "NUMTRADES")
    return _iss_history_paged(session, url, {
        "from": start, "till": end,
        "history.columns": ",".join(cols),
    })


def get_brent_front_month(session, start: str, end: str,
                          assetcode: str = "BR") -> pd.DataFrame:
    """
    Непрерывный ряд цены Brent: на каждый торговый день берём контракт BR
    с максимальным дневным объёмом (это всегда ближний ликвидный фьючерс).
    """
    frames = []
    for code in _forts_contract_codes(assetcode, start, end):
        df = get_futures_contract_history(session, code, start, end)
        if not df.empty:
            frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    allc["VOLUME"] = pd.to_numeric(allc["VOLUME"], errors="coerce").fillna(0)
    allc["CLOSE"] = pd.to_numeric(allc["CLOSE"], errors="coerce")
    allc = allc.dropna(subset=["CLOSE"])
    # На каждую дату — строка контракта с наибольшим объёмом.
    idx = allc.groupby("TRADEDATE")["VOLUME"].idxmax()
    front = allc.loc[idx].sort_values("TRADEDATE").reset_index(drop=True)
    front = front.rename(columns={"SECID": "FRONT_CONTRACT", "CLOSE": "BRENT_USD"})
    return front[["TRADEDATE", "FRONT_CONTRACT", "BRENT_USD", "VOLUME", "OPENPOSITION"]]


# =========================================================================
#  1.f  Срочный рынок: фьючерс + цепочка опционов на один день
# =========================================================================
def get_forts_futures_on_date(session, assetcode: str, date: str) -> pd.DataFrame:
    url = f"{ISS}/history/engines/futures/markets/forts/securities.json"
    df = _iss_history_paged(session, url, {
        "date": date, "assetcode": assetcode,
    })
    # Отбрасываем календарные спреды (SECID длиной 8, склейка двух контрактов).
    df = df[df["SECID"].str.len() <= 6].reset_index(drop=True)
    return df


def get_forts_options_on_date(session, assetcode: str, date: str) -> pd.DataFrame:
    url = f"{ISS}/history/engines/futures/markets/options/securities.json"
    return _iss_history_paged(session, url, {
        "date": date, "assetcode": assetcode,
    })


def get_option_specs(session, secids: Iterable[str]) -> pd.DataFrame:
    """Для каждого опциона забирает страйк, тип (C/P) и базовый фьючерс."""
    rows = []
    for secid in secids:
        payload = _iss_get(session, f"{ISS}/securities/{secid}.json")
        d = {row[0]: row[2] for row in payload["description"]["data"]}
        rows.append({
            "SECID": secid,
            "STRIKE": d.get("STRIKE"),
            "OPTIONTYPE": d.get("OPTIONTYPE"),
            "UNDERLYINGASSET": d.get("UNDERLYINGASSET"),
            "LSTDELDATE": d.get("LSTDELDATE"),
            "LSTTRADE": d.get("LSTTRADE"),
        })
    return pd.DataFrame(rows)


def get_futures_specs(session, secids: Iterable[str]) -> pd.DataFrame:
    """Для каждого фьючерса забирает дату экспирации (LSTDELDATE)."""
    rows = []
    for secid in secids:
        payload = _iss_get(session, f"{ISS}/securities/{secid}.json")
        d = {row[0]: row[2] for row in payload["description"]["data"]}
        rows.append({
            "SECID": secid,
            "LSTDELDATE": d.get("LSTDELDATE"),
            "LSTTRADE": d.get("LSTTRADE"),
            "ASSETCODE": d.get("ASSETCODE"),
        })
    df = pd.DataFrame(rows)
    df["LSTDELDATE"] = pd.to_datetime(df["LSTDELDATE"], errors="coerce")
    return df


def pick_front_future(specs: pd.DataFrame, trade_day: str, min_days: int) -> str:
    """
    Выбирает ближайший фьючерс, экспирация которого отстоит от торгового дня
    не менее чем на min_days (требование «не ближе, чем 1 месяц»).
    """
    cutoff = pd.Timestamp(trade_day) + pd.Timedelta(days=min_days)
    ok = specs[specs["LSTDELDATE"] >= cutoff].sort_values("LSTDELDATE")
    if ok.empty:
        raise ValueError("Не найден фьючерс с экспирацией не ближе заданного срока")
    return ok.iloc[0]["SECID"]


# =========================================================================
#  1.a  КРИВАЯ БЕСКУПОННОЙ ДОХОДНОСТИ (ЦБ РФ)
# =========================================================================
def _cbr_kbd_chunk(session, frm: str, to: str) -> pd.DataFrame:
    """Одна страница КБД ЦБ (даты в формате ДД.ММ.ГГГГ)."""
    url = "https://www.cbr.ru/hd_base/zcyc_params/"
    params = {"UniDbQuery.Posted": "True", "UniDbQuery.From": frm, "UniDbQuery.To": to}
    r = session.get(url, params=params, timeout=60)
    # Страница отдаётся в UTF-8 (charset объявлен в заголовке) — не переопределяем.
    tables = pd.read_html(io.StringIO(r.text), decimal=",", thousands="\xa0")
    if not tables:
        return pd.DataFrame()
    t = max(tables, key=lambda x: x.shape[0])
    # Колонки приходят как MultiIndex ('Срок до погашения, лет', '0,25') — выпрямляем.
    new_cols = []
    for c in t.columns:
        label = c[1] if isinstance(c, tuple) else c
        new_cols.append("DATE" if str(label).startswith("Дата") else str(label))
    t.columns = new_cols
    return t


def get_cbr_zcyc(session, start: str, end: str) -> pd.DataFrame:
    """
    Кривая бескупонной доходности гособлигаций (КБД) ЦБ РФ — пункт 1.a.
    Доходности (% годовых) по срокам 0.25 … 30 лет. Качаем по годам и склеиваем.
    """
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    frames = []
    year = s.year
    while year <= e.year:
        frm = max(s, dt.date(year, 1, 1)).strftime("%d.%m.%Y")
        to = min(e, dt.date(year, 12, 31)).strftime("%d.%m.%Y")
        chunk = _cbr_kbd_chunk(session, frm, to)
        if not chunk.empty:
            frames.append(chunk)
        year += 1
    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d.%m.%Y")
    df = df.drop_duplicates(subset="DATE").sort_values("DATE").reset_index(drop=True)
    # Числовые колонки сроков -> float
    for c in df.columns:
        if c != "DATE":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================================================================
#  1.e  КУРСЫ ВАЛЮТ (ЦБ РФ, официальные)
# =========================================================================
def get_cbr_fx(session, val_code: str, start: str, end: str) -> pd.DataFrame:
    """Официальный курс валюты ЦБ РФ за период (сервис XML_dynamic)."""
    d1 = dt.date.fromisoformat(start).strftime("%d/%m/%Y")
    d2 = dt.date.fromisoformat(end).strftime("%d/%m/%Y")
    url = ("https://www.cbr.ru/scripts/XML_dynamic.asp"
           f"?date_req1={d1}&date_req2={d2}&VAL_NM_RQ={val_code}")
    r = session.get(url, timeout=60)
    root = ET.fromstring(r.content)  # ответ в cp1251, парсим из байтов
    rows = []
    for rec in root.findall("Record"):
        date = rec.get("Date")
        nominal = float(rec.find("Nominal").text.replace(",", "."))
        value = float(rec.find("Value").text.replace(",", "."))
        rows.append({"DATE": date, "RATE": value / nominal})  # курс за 1 единицу
    df = pd.DataFrame(rows)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d.%m.%Y")
    return df.sort_values("DATE").reset_index(drop=True)
