
# -*- coding: utf-8 -*-
"""
Pyscript: NED MLR + ENTSO-E (direct REST API), kwartier-resolutie
- Geen HACS entsoe integratie
- Pure-Python OLS
- Services (Actions):
  - pyscript.ned_mlr_train
  - pyscript.ned_mlr_predict_7d
  - pyscript.ned_mlr_backtest
"""

import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import requests
import re
from builtins import open


# -------------------------------
# Forecast archive persistence
# -------------------------------

FORECAST_ARCHIVE_FILE = "/config/pyscript/forecast_archive.json"
_forecast_archive_days = {}

def _load_forecast_archive():
    """Laad forecast archive vanaf disk (async‑safe)."""
    global _forecast_archive_days
    try:
        with task.executor(open, FORECAST_ARCHIVE_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _forecast_archive_days = data
            else:
                _forecast_archive_days = {}
        log.info("Forecast archive geladen vanaf disk")
    except FileNotFoundError:
        _forecast_archive_days = {}
        log.info("Forecast archive bestand bestaat nog niet")
    except Exception as e:
        log.error(f"Fout bij laden forecast archive: {e}")
        _forecast_archive_days = {}


def _save_forecast_archive():
    """Schrijf forecast archive naar disk (async‑safe)."""
    try:
        with task.executor(open, FORECAST_ARCHIVE_FILE, "w") as f:
            json.dump(_forecast_archive_days, f)
        log.info("Forecast archive opgeslagen naar disk")
    except Exception as e:
        log.error(f"Fout bij opslaan forecast archive: {e}")

# -------------------------------
# backtest results persistence
# -------------------------------
BACKTEST_FILE = "/config/pyscript/backtest_results.json"
_backtest_results = {}

def _load_backtest_results():
    global _backtest_results
    try:
        with task.executor(open, BACKTEST_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _backtest_results = data
            else:
                _backtest_results = {}
    except FileNotFoundError:
        _backtest_results = {}
    except Exception as e:
        log.error(f"Fout bij laden backtest results: {e}")
        _backtest_results = {}

def _save_backtest_results():
    try:
        with task.executor(open, BACKTEST_FILE, "w") as f:
            json.dump(_backtest_results, f)
    except Exception as e:
        log.error(f"Fout bij opslaan backtest results: {e}")


# -------------------------------
# Coefficients persistence
# -------------------------------
COEF_FILE = "/config/pyscript/mlr_coefficients.json"
_coef_data = {"coef": None, "meta": {}}

def _load_coef():
    global _coef_data
    try:
        with task.executor(open, COEF_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _coef_data = data
    except FileNotFoundError:
        _coef_data = {"coef": None, "meta": {}}
    except Exception as e:
        log.error(f"Fout bij laden coef: {e}")
        _coef_data = {"coef": None, "meta": {}}

def _save_coef(coef, meta: dict):
    global _coef_data
    _coef_data = {"coef": coef, "meta": meta}
    try:
        with task.executor(open, COEF_FILE, "w") as f:
            json.dump(_coef_data, f)
    except Exception as e:
        log.error(f"Fout bij opslaan coef: {e}")

def _get_coef():
    """Return coef list or None if not trained yet."""
    return _coef_data.get("coef")

@time_trigger("startup")
def _load_coef_on_startup():
    _load_coef()

# -------------------------------
# MQTT helpers (discovery + state)
# -------------------------------

MQTT_DISCOVERY_PREFIX = "homeassistant"

MQTT_SENSOR_COEF_ID = "ned_mlr_coeff"
MQTT_SENSOR_COEF_NAME = "NED MLR Coefficients"
MQTT_SENSOR_COEF_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_COEF_ID}/config"
MQTT_SENSOR_COEF_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_COEF_ID}/state"

MQTT_SENSOR_PROGRESS_ID = "ned_mlr_train_progress"
MQTT_SENSOR_PROGRESS_NAME = "NED MLR Train Progress"
MQTT_SENSOR_PROGRESS_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_PROGRESS_ID}/config"
MQTT_SENSOR_PROGRESS_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_PROGRESS_ID}/state"

MQTT_SENSOR_FORECAST_ID = "ned_mlr_price_forecast"
MQTT_SENSOR_FORECAST_NAME = "NED MLR Price Forecast"
MQTT_SENSOR_FORECAST_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECAST_ID}/config"
MQTT_SENSOR_FORECAST_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECAST_ID}/state"

MQTT_SENSOR_BACKTEST_ID = "ned_mlr_backtest"
MQTT_SENSOR_BACKTEST_NAME = "NED MLR Backtest"
MQTT_SENSOR_BACKTEST_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_BACKTEST_ID}/config"
MQTT_SENSOR_BACKTEST_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_BACKTEST_ID}/state"

MQTT_SENSOR_FORECASTBAND_ID = "ned_mlr_price_forecast_band"
MQTT_SENSOR_FORECASTBAND_NAME = "NED MLR Price Forecast Band"
MQTT_SENSOR_FORECASTBAND_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECASTBAND_ID}/config"
MQTT_SENSOR_FORECASTBAND_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECASTBAND_ID}/state"

MQTT_SENSOR_FORECASTARCHIVE_ID = "ned_mlr_forecast_archive"
MQTT_SENSOR_FORECASTARCHIVE_NAME = "NED MLR Forecast Archive"
MQTT_SENSOR_FORECASTARCHIVE_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECASTARCHIVE_ID}/config"
MQTT_SENSOR_FORECASTARCHIVE_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_FORECASTARCHIVE_ID}/state"

MQTT_SENSOR_BACKTESTFORECASTED_ID = "ned_mlr_backtest_forecasted"
MQTT_SENSOR_BACKTESTFORECASTED_NAME = "NED MLR Backtest Forecasted"
MQTT_SENSOR_BACKTESTFORECASTED_CONFIG_TOPIC = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_BACKTESTFORECASTED_ID}/config"
MQTT_SENSOR_BACKTESTFORECASTED_STATE_TOPIC  = f"{MQTT_DISCOVERY_PREFIX}/sensor/{MQTT_SENSOR_BACKTESTFORECASTED_ID}/state"

_discovery_published = False


def _publish_mqtt_discovery():
    """
    Publish MQTT Discovery config for all sensors (once per HA startup).
    """
    global _discovery_published
    if _discovery_published:
        return

    # 1 Coefficients sensor
    mqtt.publish(
        topic=MQTT_SENSOR_COEF_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_COEF_NAME,
            "unique_id": MQTT_SENSOR_COEF_ID,
            "state_topic": MQTT_SENSOR_COEF_STATE_TOPIC,
            "icon": "mdi:chart-line",
            # state is JSON; extract 'value' as main state
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_COEF_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 2 Progress sensor
    mqtt.publish(
        topic=MQTT_SENSOR_PROGRESS_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_PROGRESS_NAME,
            "unique_id": MQTT_SENSOR_PROGRESS_ID,
            "state_topic": MQTT_SENSOR_PROGRESS_STATE_TOPIC,
            "icon": "mdi:progress-clock",
            # state is JSON; extract 'status' as main state
            "value_template": "{{ value_json.status | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_PROGRESS_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 3 Price forecast sensor
    mqtt.publish(
        topic=MQTT_SENSOR_FORECAST_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_FORECAST_NAME,
            "unique_id": MQTT_SENSOR_FORECAST_ID,
            "state_topic": MQTT_SENSOR_FORECAST_STATE_TOPIC,
            "icon": "mdi:chart-bell-curve",
            # The main state is the numeric forecast value
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_FORECAST_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 4 Backtest sensor
    mqtt.publish(
        topic=MQTT_SENSOR_BACKTEST_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_BACKTEST_NAME,
            "unique_id": MQTT_SENSOR_BACKTEST_ID,
            "state_topic": MQTT_SENSOR_BACKTEST_STATE_TOPIC,
            "icon": "mdi:chart-line",
            # state is JSON; extract 'value' as main state
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_BACKTEST_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 5 NED MLR Price Forecast Band
    mqtt.publish(
        topic=MQTT_SENSOR_FORECASTBAND_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_FORECASTBAND_NAME,
            "unique_id": MQTT_SENSOR_FORECASTBAND_ID,
            "state_topic": MQTT_SENSOR_FORECASTBAND_STATE_TOPIC,
            "icon": "mdi:progress-clock",
            # state is JSON; extract 'status' as main state
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_FORECASTBAND_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 6 ned_mlr_forecast_archive sensor
    mqtt.publish(
        topic=MQTT_SENSOR_FORECASTARCHIVE_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_FORECASTARCHIVE_NAME,
            "unique_id": MQTT_SENSOR_FORECASTARCHIVE_ID,
            "state_topic": MQTT_SENSOR_FORECASTARCHIVE_STATE_TOPIC,
            "icon": "mdi:chart-bell-curve",
            # The main state is the numeric forecast value
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_FORECASTARCHIVE_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )

    # 7 ned_mlr_backtest forcasted sensor
    mqtt.publish(
        topic=MQTT_SENSOR_BACKTESTFORECASTED_CONFIG_TOPIC,
        payload=json.dumps({
            "name": MQTT_SENSOR_BACKTESTFORECASTED_NAME,
            "unique_id": MQTT_SENSOR_BACKTESTFORECASTED_ID,
            "state_topic": MQTT_SENSOR_BACKTESTFORECASTED_STATE_TOPIC,
            "icon": "mdi:chart-bell-curve",
            # The main state is the numeric forecast value
            "value_template": "{{ value_json.value | default('unknown') }}",
            "json_attributes_topic": MQTT_SENSOR_BACKTESTFORECASTED_STATE_TOPIC,
        }),
        qos=1,
        retain=True,
    )    

    _discovery_published = True
    log.info("NED MLR: MQTT discovery config published")


@time_trigger("startup")
def ned_mlr_mqtt_discovery_startup():
    """
    Publish MQTT discovery config once at HA startup.
    """
    try:
        _publish_mqtt_discovery()
    except Exception as e:
        log.error(f"NED MLR: MQTT discovery at startup failed: {e}")

@time_trigger("startup")
def _load_archive_on_startup():
    _load_forecast_archive()

@time_trigger("startup")
def _load_backtest_on_startup():
    _load_backtest_results()    

def _mqtt_publish_coefficients(coef, meta: dict):
    """
    Publish coefficients + metadata to MQTT state topic.
    """
    payload = {
        "value": coef,
    }
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_COEF_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )


def _mqtt_publish_progress(status: str, meta: dict = None):
    """
    Publish training progress + metadata to MQTT state topic.
    'status' is a short string: started, running, done, error, etc.
    """
    payload = {
        "status": status,
    }
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_PROGRESS_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )

def _mqtt_publish_forecast(value, meta: dict = None):
    """
    Publish price forecast + attributes to MQTT.
    """
    payload = {"value": value}
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_FORECAST_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )    

def _mqtt_publish_backtest(value, meta: dict = None):
    """
    Publish price backtest + attributes to MQTT.
    """
    payload = {"value": value}
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_BACKTEST_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )    

def _mqtt_publish_forecast_band(value, meta: dict = None):
    """
    Publish price forecast band+ attributes to MQTT.
    """
    payload = {"value": value}
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_FORECASTBAND_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )    

def _mqtt_publish_forecast_archive(value, meta: dict = None):
    """
    Publish price forecast archive + attributes to MQTT.
    """
    payload = {"value": value}
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_FORECASTARCHIVE_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )    

def _mqtt_publish_backtest_forecasted(value, meta: dict = None):
    """
    Publish price forecast archive + attributes to MQTT.
    """
    payload = {"value": value}
    if meta:
        payload.update(meta)

    mqtt.publish(
        topic=MQTT_SENSOR_BACKTESTFORECASTED_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )    

# -------------------------------
# Config uit Pyscript apps
# -------------------------------
APP = pyscript.app_config or {}
NED_KEY     = APP.get("ned_api_key", "")
ENTSOE_KEY  = APP.get("entsoe_api_key", "")
POINT_NL    = int(APP.get("point_nl", 0))  # 0 -> NL autodetect fallback

# -------------------------------
# Endpoints & constants
# -------------------------------
NED_BASE    = "https://api.ned.nl/v1"
ENTSOE_API  = "https://web-api.tp.entsoe.eu/api"
EIC_NL      = "10YNL----------L"  # NL bidding zone (EIC)

HEADERS_JSONLD = {"Accept": "application/ld+json", "X-AUTH-TOKEN": NED_KEY}

SOLAR_TYPE_ID       = 2
WIND_ON_TYPE_ID     = 1
WIND_OFF_TYPE_ID    = 17
CONSUMPTION_TYPE_ID = 59

# -------------------------------
# Helpers zonder generator-expressions
# -------------------------------
def _parse_ts_any_to_local(ts_str: str):
    """
    Parse een timestampstring (met of zonder offset / met 'Z' / met '+0100')
    en retourneer een timezone-aware datetime in Europe/Amsterdam (seconds, geen microseconds).
    """
    if not ts_str:
        return None
    s = ts_str.strip()

    # 'Z' -> '+00:00'
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'

    # '+0100' / '-0530' -> '+01:00' / '-05:30'
    m = re.search(r'([+-])(\d{2})(\d{2})$', s)
    if m and (':' not in s[-6:]):  # laatste 6 chars hebben geen ':'
        s = s[:-5] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"

    # parse als ISO, zo niet: fallback zonder offset
    dt = None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

    # naar Europe/Amsterdam
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Amsterdam"))
    else:
        dt = dt.astimezone(ZoneInfo("Europe/Amsterdam"))

    return dt.replace(microsecond=0)

def _canonicalize_dict_keys(d: dict) -> dict:
    """
    Zet alle keys van {ts_str -> value} om naar uniforme ISO strings
    (Europe/Amsterdam, seconds, mét offset).
    """
    out = {}
    for k, v in d.items():
        dt = _parse_ts_any_to_local(k)
        if dt is not None:
            out[dt.isoformat(timespec='seconds')] = v
    return out

def contains_any(text, keywords):
    """Return True als één van de keywords in text zit (case-insensitive downstream al toegepast)."""
    for k in keywords:
        if k in text:
            return True
    return False


def _metrics_no_gen(y_true, y_pred):
    """
    Bereken MAE, RMSE, R2 zonder generator expressions (Pyscript‑safe).
    """
    n = len(y_true)
    if n == 0:
        return {"mae": None, "rmse": None, "r2": None}

    # MAE
    s_abs = 0.0
    for i in range(n):
        s_abs += abs(y_true[i] - y_pred[i])
    mae = s_abs / n

    # RMSE
    s_sq = 0.0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        s_sq += diff * diff
    rmse = (s_sq / n) ** 0.5

    # R²
    s_true = 0.0
    for i in range(n):
        s_true += y_true[i]
    mean_y = s_true / n

    ss_tot = 0.0
    for i in range(n):
        d = y_true[i] - mean_y
        ss_tot += d * d

    ss_res = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        ss_res += d * d

    r2 = None
    if ss_tot > 1e-12:
        r2 = 1.0 - (ss_res / ss_tot)

    return {"mae": mae, "rmse": rmse, "r2": r2}

# -------------------------------
# Pure-Python OLS (geen scikit)
# -------------------------------
def _matmul(A, B):
    m, n = len(A), len(A[0])
    assert len(B) == n
    p = len(B[0])
    C = [[0.0]*p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(p):
                C[i][j] += aik * Bk[j]
    return C

def _transpose(M):
    return [list(row) for row in zip(*M)]

def _gauss_jordan(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        maxabs = abs(M[pivot][col])
        for r in range(col+1, n):
            v = abs(M[r][col])
            if v > maxabs:
                maxabs = v; pivot = r
        if maxabs < 1e-12:
            raise ValueError(f"Singulier XtX (col {col})")
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        pivval = M[col][col]
        invp = 1.0 / pivval
        for j in range(col, n+1):
            M[col][j] *= invp
        for r in range(n):
            if r == col: 
                continue
            fac = M[r][col]
            if fac != 0.0:
                for j in range(col, n+1):
                    M[r][j] -= fac * M[col][j]
    return [M[i][n] for i in range(n)]

def ols_fit(X, y, add_intercept=True):
    n = len(X)
    if n == 0:
        raise ValueError("Lege X")
    Xw = []
    if add_intercept:
        for row in X:
            Xw.append([1.0] + [float(v) for v in row])
    else:
        for row in X:
            Xw.append([float(v) for v in row])
    Xt  = _transpose(Xw)
    XtX = _matmul(Xt, Xw)
    XtY = []
    for i in range(len(Xt)):
        s = 0.0
        Xi = Xt[i]
        for k in range(n):
            s += Xi[k] * float(y[k])
        XtY.append(s)
    beta = _gauss_jordan(XtX, XtY)
    return beta

def ols_predict(X, coef, add_intercept=True):
    out = []
    for row in X:
        s = 0.0
        idx = 0
        if add_intercept:
            s += coef[0]; idx = 1
        for j, v in enumerate(row):
            s += coef[idx + j] * float(v)
        out.append(s)
    return out

# -------------------------------
# NED helpers & discovery
# -------------------------------
def _iri_to_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip().split('/')[-1])
        except Exception:
            return None
    return None

def _paged_get(url, params, max_pages=40, delay_s=0.25):
    items = []
    for page in range(1, max_pages+1):
        p = dict(params)
        p["page"] = page
        # blocking I/O via executor
        resp = task.executor(requests.get, url, params=p, headers=HEADERS_JSONLD, allow_redirects=False, timeout=30)
        if resp.status_code in (400, 403):
            break
        if resp.status_code == 429:
            task.sleep(0.7 + random.uniform(0, 0.5))
            continue
        resp.raise_for_status()
        data = resp.json()
        member = data.get("hydra:member", [])
        if not member:
            break
        items.extend(member)
        task.sleep(delay_s)
    return items

def _ned_get(path):
    url = f"{NED_BASE}/{path}"
    resp = task.executor(requests.get, url, headers=HEADERS_JSONLD, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hydra:member", [])

def find_ned_ids():
    if not NED_KEY:
        log.error("NED key ontbreekt (apps-config).")
        return None
    # points
    pts = _ned_get("points")
    pid_nl = None
    for p in pts:
        name = (p.get("name") or "").lower()
        pid  = _iri_to_int(p.get("id") or p.get("@id") or p.get("pointId"))
        if "nederland" in name or "netherlands" in name:
            pid_nl = pid; break
    if pid_nl is None and pts:
        pid_nl = _iri_to_int(pts[0].get("id"))

    # timezones
    tzs = _ned_get("granularity_time_zones")
    tz_id = None
    for tz in tzs:
        nm = (tz.get("name") or "").lower()
        if "amsterdam" in nm:
            tz_id = _iri_to_int(tz.get("id")); break
    if tz_id is None and tzs:
        tz_id = _iri_to_int(tzs[0].get("id"))

    # granularities
    gns = _ned_get("granularities")
    gran15 = gran60 = None
    for g in gns:
        nm = (g.get("name") or "").lower()
        gid = _iri_to_int(g.get("id"))
        if contains_any(nm, ["15", "pt15m", "kwartier"]):
            gran15 = gid
        if contains_any(nm, ["60", "pt60m", "uur", "hour"]):
            gran60 = gid

    # activities
    acts = _ned_get("activities")
    act_prov = act_cons = None
    for a in acts:
        nm = (a.get("name") or "").lower()
        aid = _iri_to_int(a.get("id"))
        if contains_any(nm, ["provid", "product", "opwek"]):
            act_prov = aid
        if contains_any(nm, ["consum", "verbruik", "afname", "demand", "load", "vraag"]):
            act_cons = aid
    if act_prov is None and acts:
        act_prov = _iri_to_int(acts[0].get("id"))

    # classifications
    cls = _ned_get("classifications")
    cls_cur = cls_back = cls_fore = None
    for c in cls:
        nm = (c.get("name") or "").lower()
        cid = _iri_to_int(c.get("id"))
        if "current" in nm:
            cls_cur  = cid
        elif "backcast" in nm:
            cls_back = cid
        elif contains_any(nm, ["forecast", "verwachting", "voorspelling"]):
            cls_fore = cid

    return {
        "point_nl": pid_nl,
        "tz_id": tz_id,
        "gran15": gran15,
        "gran60": gran60,
        "act_prov": act_prov,
        "act_cons": act_cons,
        "cls_cur": cls_cur,
        "cls_back": cls_back,
        "cls_fore": cls_fore,
    }

def _items_to_rows(items):
    rows = []
    for it in items:
        ts = it.get("validfrom") or it.get("validFrom")
        if not ts:
            continue
        val = None
        for key in ("volume","value","capacity","percentage"):
            if key in it and it.get(key) is not None:
                try:
                    val = float(str(it[key]).replace(",", "."))
                    break
                except Exception:
                    pass
        if val is None:
            continue
        rows.append((ts, val))
    rows.sort(key=lambda r: r[0])
    return rows

def fetch_ned_q15_range(start_date, end_date, type_id, activity_id, granularity_id, tz_id, point_ids, classification_id):
    """
    Haal NED-utilizations op en return dict ts->value (Europe/Amsterdam; kwartier of upsample later)
    """
    out = {}
    sdt = datetime.fromisoformat(start_date)
    edt = datetime.fromisoformat(end_date)
    cur = sdt
    while cur < edt:
        chunk_end = cur + timedelta(days=1)
        for pt in point_ids:
            params = {
                "point": pt,
                "granularity": granularity_id,
                "granularitytimezone": tz_id,
                "classification": classification_id,
                "activity": activity_id,
                "type": type_id,
                "validfrom[strictly_before]": chunk_end.date().isoformat(),
                "validfrom[after]": cur.date().isoformat(),
                "itemsPerPage": 1000,
            }
            items = _paged_get(f"{NED_BASE}/utilizations", params, max_pages=40, delay_s=0.2)
            rows = _items_to_rows(items)
            for ts, v in rows:
                out[ts] = out.get(ts, 0.0) + v
        cur = chunk_end
    return out  # ts in Europe/Amsterdam (NED levert local ISO)

def _upsample_hour_dict_to_q15_local(hour_dict):
    """
    Input: dict ts_local_iso (Europe/Amsterdam) per uur -> waarde
    Output: dict ts_local_iso per kwartier -> waarde (ffill)
    """
    q15 = {}
    for ts in sorted(hour_dict.keys()):
        dt = datetime.fromisoformat(ts)
        val = hour_dict[ts]
        for k in range(4):
            tsq = (dt + timedelta(minutes=15*k)).isoformat()
            q15[tsq] = val
    return q15

# -------------------------------
# ENTSO-E REST API (direct)
# -------------------------------
def _to_utc_yyyymmddhhmm(dt_local):
    return dt_local.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")


def _local_tag(tag: str) -> str:
    """Haal het 'local' deel van een XML tag (na '}' in {ns}tag)."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def _parse_publication_marketdocument(xml_text):
    """
    Parse ENTSO-E Publication_MarketDocument en retourneer
    lijst van (start_iso_utc, resolution_str, [(position, price)]).
    Robuust voor tag 'price.amount' met punt in naam.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        log.error(f"ENTSO-E XML parse error: {e}")
        return []

    # Zoek periode-interval (start)
    period_interval = None
    for el in root.iter():
        if _local_tag(el.tag) == "period.timeInterval":
            period_interval = el
            break
    if period_interval is None:
        return []

    start_el = None
    for ch in period_interval:
        if _local_tag(ch.tag) == "start":
            start_el = ch
            break
    if start_el is None or not start_el.text:
        return []
    start = start_el.text  # bv. '2025-12-31T23:00Z'

    series = []
    # Vind alle TimeSeries
    for ts in root.iter():
        if _local_tag(ts.tag) != "TimeSeries":
            continue
        # Zoek Period + resolution
        period_el = None
        for ch in ts.iter():
            if _local_tag(ch.tag) == "Period":
                period_el = ch
                break
        if period_el is None:
            continue

        resolution = "PT60M"
        for ch in period_el:
            if _local_tag(ch.tag) == "resolution" and ch.text:
                resolution = ch.text.strip()
                break

        points = []
        # Alle Point entries met position & price.amount
        for pt in period_el:
            if _local_tag(pt.tag) != "Point":
                continue
            pos = None
            price = None
            for child in pt:
                ltag = _local_tag(child.tag)
                if ltag == "position":
                    try:
                        pos = int(child.text)
                    except Exception:
                        pos = None
                elif ltag == "price.amount":  # let op punt in tagnaam
                    try:
                        price = float(child.text)
                    except Exception:
                        price = None
            if pos is not None and price is not None:
                points.append((pos, price))

        if points:
            series.append((start, resolution, points))

    return series


def _series_to_q15_local(series_list):
    """
    Zet ENTSO-E (UTC) series om naar Europe/Amsterdam kwartieren.
    """
    out = {}
    for start_iso_utc, res, points in series_list:
        dt0_utc = datetime.fromisoformat(start_iso_utc.replace("Z", "+00:00"))
        if res == "PT15M":
            step = timedelta(minutes=15); quarters = 1
        elif res == "PT30M":
            step = timedelta(minutes=30); quarters = 2
        else:
            step = timedelta(hours=1); quarters = 4
        for pos, price in points:
            base_utc = dt0_utc + step * (pos - 1)
            for k in range(quarters):
                ts_local = (base_utc + timedelta(minutes=15*k)).astimezone(ZoneInfo("Europe/Amsterdam")).isoformat()
                out[ts_local] = price
    return out


def _entsoe_call(params, max_retries=6, base_sleep=0.6, hard_cap=8.0):
    """
    Eén HTTP-call met exponential backoff + jitter.
    Retourneert response.text (str) of None bij definitieve fout.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = task.executor(requests.get, ENTSOE_API, params=params, timeout=30)
            # 200 OK
            if resp.status_code == 200:
                return resp.text
            # 429 / 5xx -> backoff
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                wait = min((base_sleep * (2 ** attempt)) + random.uniform(0, 0.4), hard_cap)
                log.warning(f"ENTSO-E {resp.status_code}, retry in {wait:.2f}s (attempt {attempt+1}/{max_retries})")
                task.sleep(wait)
                continue
            # 4xx anders: geen retry
            log.error(f"ENTSO-E HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as e:
            last_err = e
            wait = min((base_sleep * (2 ** attempt)) + random.uniform(0, 0.4), hard_cap)
            log.warning(f"ENTSO-E connect error ({e}); retry in {wait:.2f}s (attempt {attempt+1}/{max_retries})")
            task.sleep(wait)
    if last_err:
        log.error(f"ENTSO-E definitieve fout: {last_err}")
    return None


def fetch_entsoe_prices_q15_range_api(start_date: str, end_date: str, eic_code: str = EIC_NL) -> dict:
    """
    Directe ENTSO-E call voor A44/A01 prijzen, dag-voor-dag, met retry/backoff.
    Normaliseert naar kwartier en Europe/Amsterdam. Slaat mislukte dagen over.
    """
    if not ENTSOE_KEY:
        log.error("ENTSO-E key ontbreekt (apps-config).")
        return {}
    sday = datetime.fromisoformat(start_date).date()
    eday = datetime.fromisoformat(end_date).date()
    out = {}
    cur = sday
    while cur < eday:
        day_local_start = datetime.combine(cur, dtime(0,0,0), tzinfo=ZoneInfo("Europe/Amsterdam"))
        day_local_end   = day_local_start + timedelta(days=1)
        params = {
            "securityToken": ENTSOE_KEY,
            "documentType": "A44",  # Price doc
            "processType":  "A01",  # Day-ahead
            "in_Domain": eic_code,
            "out_Domain": eic_code,
            "periodStart": _to_utc_yyyymmddhhmm(day_local_start),
            "periodEnd":   _to_utc_yyyymmddhhmm(day_local_end),
        }
        xml_text = _entsoe_call(params)
        if xml_text:
            series_list = _parse_publication_marketdocument(xml_text)
            if series_list:
                q15_local = _series_to_q15_local(series_list)
                out.update(q15_local)
            else:
                log.warning(f"ENTSO-E: geen prijzen in XML voor {cur}")
        else:
            log.warning(f"ENTSO-E: dag {cur} overgeslagen (geen verbinding/HTTP-error)")
        task.sleep(0.15)  # zacht throttle
        cur = cur + timedelta(days=1)
    return out


# -------------------------------
# Feature builder (kwartier + lag1)
# -------------------------------
def build_feature_matrix_q15(solar, won, wof, cons):
    # Alle dicts: ts_local_iso -> value
    keys = sorted(set(solar) & set(won) & set(wof) & set(cons))
    X, idx = [], []
    for ts in keys:
        X.append([solar[ts], won[ts], wof[ts], cons[ts]])
        idx.append(ts)
    # lag1
    Xlag = []
    for i in range(len(X)):
        if i == 0:
            Xlag.append(X[i] + [0.0,0.0,0.0,0.0])
        else:
            prev = X[i-1]
            Xlag.append(X[i] + prev)
    return idx, Xlag


# -------------------------------
# forcast archive builder
# -------------------------------
def _build_forecast_archive_days(
    anchor_day,
    coef,
    hist_solar, hist_won, hist_wof, hist_cons,
    tz_id, gran15, gran60,
    act_prov, act_cons, cls_fore,
    point_nl,
    H,
    latest_hist
):
    """
    Bouwt forecast-archief per dag, in EUR/MWh.
    - anchor_day: date()
    - H: max horizon (int)
    - latest_hist: laatste dag met ENTSO-E historie (date)
    """

    archive_by_day = {}

    # Zorg dat anchor_day een date is
    if isinstance(anchor_day, datetime):
        anchor_date = anchor_day.date()
    else:
        anchor_date = anchor_day

    # Forecast range
    sdate = anchor_date.isoformat()
    max_end = min(anchor_date + timedelta(days=H), latest_hist + timedelta(days=1))
    edate = max_end.isoformat()

    # Forecast features
    solar15 = fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_fore)
    won15   = fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
    wof15   = fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
    cons15  = fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_fore) if act_cons else {}

    # fallback 60m
    if not solar15 and gran60:
        solar15 = _upsample_hour_dict_to_q15_local(
            fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran60, tz_id, [point_nl], cls_fore)
        )
    if not won15 and gran60:
        won15 = _upsample_hour_dict_to_q15_local(
            fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore)
        )
    if not wof15 and gran60:
        wof15 = _upsample_hour_dict_to_q15_local(
            fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore)
        )
    if not cons15 and gran60 and act_cons:
        cons15 = _upsample_hour_dict_to_q15_local(
            fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran60, tz_id, [point_nl], cls_fore)
        )

    if not solar15 or not won15 or not wof15:
        return {}

    if not cons15:
        cons15 = {}
        keys_union = set()
        for d in (solar15, won15, wof15):
            for k in d.keys():
                keys_union.add(k)
        for k in keys_union:
            cons15[k] = 0.0

    # Normaliseer keys
    solar15 = _canonicalize_dict_keys(solar15)
    won15   = _canonicalize_dict_keys(won15)
    wof15   = _canonicalize_dict_keys(wof15)
    cons15  = _canonicalize_dict_keys(cons15)

    # Feature matrix
    f_idx, Xf = build_feature_matrix_q15(solar15, won15, wof15, cons15)

    # Lag seed
    def _last(d):
        if not d:
            return 0.0
        ks = list(d.keys())
        ks.sort()
        return d[ks[-1]]

    seed = [
        _last(hist_solar),
        _last(hist_won),
        _last(hist_wof),
        _last(hist_cons) if hist_cons else 0.0
    ]

    for i in range(len(Xf)):
        if i == 0:
            for j in range(4, len(Xf[i])):
                Xf[i][j] = seed[j - 4]
        else:
            for j in range(4, len(Xf[i])):
                Xf[i][j] = Xf[i - 1][j - 4]

    # Voorspelling (EUR/MWh)
    yhat = ols_predict(Xf, coef, add_intercept=True)

    now_iso = datetime.now().isoformat(timespec="seconds")

    for i in range(len(f_idx)):
        ts = f_idx[i]
        dt_start = datetime.fromisoformat(ts)
        dt_end   = dt_start + timedelta(minutes=15)

        local_start = dt_start.isoformat()
        local_end   = dt_end.isoformat()

        center_eur_mwh = float(yhat[i])

        pred_date = dt_start.date()
        horizon = (pred_date - anchor_date).days + 1
        if horizon < 1:
            horizon = 1
        if horizon > H:
            horizon = H

        day_key = pred_date.isoformat()
        if day_key not in archive_by_day:
            archive_by_day[day_key] = {
                "generated_at": now_iso,
                "data": []
            }

        archive_by_day[day_key]["data"].append({
            "start": local_start,
            "end": local_end,
            "center": center_eur_mwh,
            "horizon": horizon
        })

    return archive_by_day

def archive_add_forecast(day_key, anchor_day, horizon, entries):
    global _forecast_archive_days

    # Dag bestaat nog niet → maak volledige structuur
    if day_key not in _forecast_archive_days:
        _forecast_archive_days[day_key] = {"forecasts": []}

    # Dag bestaat wel maar zonder forecasts → maak forecasts-array
    if "forecasts" not in _forecast_archive_days[day_key]:
        _forecast_archive_days[day_key]["forecasts"] = []

    # Voeg toe
    _forecast_archive_days[day_key]["forecasts"].append({
        "anchor": anchor_day,
        "horizon": horizon,
        "data": entries
    })
    
def archive_prune(retain_days=14):
    global _forecast_archive_days

    today = datetime.now().date()
    cutoff = today - timedelta(days=retain_days)

    pruned = {}
    for dk, payload in _forecast_archive_days.items():
        try:
            ddate = datetime.fromisoformat(dk).date()
            if ddate >= cutoff:
                pruned[dk] = payload
        except:
            pass

    _forecast_archive_days = pruned
    _save_forecast_archive()

# -------------------------------
# TRAIN
# -------------------------------
@service
def ned_mlr_train(start_date: str=None, end_date: str=None):
    """
    Train MLR op kwartier (NED features + ENTSO-E prijzen direct via REST).
    """
    log.info("ned_mlr_train: start")
    ids = find_ned_ids()
    if not ids:
        return
    tz_id     = ids["tz_id"]
    gran15    = ids["gran15"]
    act_prov  = ids["act_prov"]
    act_cons  = ids["act_cons"]
    cls_cur   = ids["cls_cur"]
    cls_back  = ids["cls_back"]
    point_nl  = POINT_NL if POINT_NL != 0 else ids["point_nl"]

    today = datetime.today().date()
    end_date   = end_date or today.isoformat()
    start_date = start_date or (today - timedelta(days=90)).isoformat()

    # NED kwartier (current/backcast; neem één classificatie die beschikbaar is)
    cls_hist = cls_cur if cls_cur is not None else cls_back
    solar = fetch_ned_q15_range(start_date, end_date, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_hist)
    won   = fetch_ned_q15_range(start_date, end_date, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist)
    wof   = fetch_ned_q15_range(start_date, end_date, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist)
    cons  = {}

    if act_cons is not None:
        cons = fetch_ned_q15_range(start_date, end_date, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_hist)
    if not cons:
        cons = {ts: 0.0 for ts in set(solar) | set(won) | set(wof)}
    
    # normalize times
    solar = _canonicalize_dict_keys(solar)
    won   = _canonicalize_dict_keys(won)
    wof   = _canonicalize_dict_keys(wof)
    cons  = _canonicalize_dict_keys(cons)

    # ENTSO-E prijzen direct via REST (kwartier + Europe/Amsterdam)
    prices_q15 = fetch_entsoe_prices_q15_range_api(start_date, end_date, eic_code=EIC_NL)
    prices_q15 = _canonicalize_dict_keys(prices_q15)
    
    # Debug: laat aantallen zien
    log.info(f"TRAIN: solar={len(solar)}, won={len(won)}, wof={len(wof)}, cons={len(cons)}, prices_q15={len(prices_q15)}")
    if prices_q15:
        # Laat 3 voorbeeld timestamps zien
        eg_ts = list(prices_q15.keys())[:3]
        log.info(f"TRAIN: sample ENTSO-E ts={eg_ts}")
    

    # Matrix + target aligneren
    idx, X = build_feature_matrix_q15(solar, won, wof, cons)
    y = [prices_q15.get(ts) for ts in idx]
    X2, y2 = [], []
    for i, yi in enumerate(y):
        if yi is not None:
            X2.append(X[i]); y2.append(yi)

    if not X2:
        log.error("Geen overlap features/prijzen; controleer ENTSO-E key of datumbereik.")
        return

    coef = ols_fit(X2, y2, add_intercept=True)

    meta = {
        "features": ["solar","wind_on","wind_off","cons","lag_solar","lag_won","lag_wof","lag_cons"],
        "start": start_date,
        "end": end_date,
    }

    _publish_mqtt_discovery()
    _save_coef(coef, meta)
    _mqtt_publish_coefficients(coef, meta)

    log.info("ned_mlr_train: klaar, coef gepubliceerd via MQTT")


@service
def ned_mlr_train_long(start_date: str = None, end_date: str = None, batch_days: int = 30):
    """
    Hybride training: splitst [start_date, end_date) in batches en traint in een background task.
    - start_date / end_date: ISO 'YYYY-MM-DD'; default: laatste 180 dagen tot vandaag.
    - batch_days: aantal dagen per batch (default 30).
    Voortgang in sensor.ned_mlr_train_progress; uiteindelijke coefs in sensor.ned_mlr_coeff.
    """
    try:
        # Valideer batch_days
        try:
            batch_days = int(batch_days)
        except Exception:
            batch_days = 30
        if batch_days < 7:
            batch_days = 7
        if batch_days > 60:
            batch_days = 60

        # Defaults voor datumbereik
        today = datetime.now(tz=ZoneInfo("Europe/Amsterdam")).date()
        if not end_date:
            end_date = today.isoformat()
        if not start_date:
            start_date = (today - timedelta(days=180)).isoformat()

        # Parse en normaliseer (half-open eind: [start, end))
        s_day = datetime.fromisoformat(start_date).date()
        e_day = datetime.fromisoformat(end_date).date()
        if e_day <= s_day:
            log.error("ned_mlr_train_long: end_date <= start_date, corrigeer datumbereik.")
            _publish_mqtt_discovery()
            _mqtt_publish_progress("error", {
                "msg": "end_date moet > start_date zijn",
                "start": start_date,
                "end": end_date,
            })
            return

        # Fire-and-forget: background task starten
        task.create(_ned_mlr_train_long_bg, s_day, e_day, batch_days)
        _publish_mqtt_discovery()
        _mqtt_publish_progress("started", {
            "start": start_date,
            "end": end_date,
            "batch_days": batch_days,
        })
        log.info(f"ned_mlr_train_long: gestart voor {start_date}..{end_date} (batch={batch_days}d)")
    except Exception as e:
        log.error(f"ned_mlr_train_long: init-fout {e}")
        _publish_mqtt_discovery()
        _mqtt_publish_progress("error", {"msg": str(e)})
        return


def _date_chunks(s_day, e_day, batch_days):
    """
    Maak lijst van (chunk_start, chunk_end) datums; half-open [start, end)
    """
    chunks = []
    cur = s_day
    while cur < e_day:
        nxt = cur + timedelta(days=batch_days)
        if nxt > e_day:
            nxt = e_day
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _safe_canon(d):
    return _canonicalize_dict_keys(d or {})


def _append_xy(X_acc, y_acc, solar, won, wof, cons, prices):
    """
    Bouw features/target per batch en append de overlappende voorbeelden aan X_acc, y_acc.
    """
    idx, X = build_feature_matrix_q15(solar, won, wof, cons)
    # Align met prijzen
    for i, ts in enumerate(idx):
        yi = prices.get(ts)
        if yi is not None:
            X_acc.append(X[i])
            y_acc.append(yi)


def _sum_len_dicts(*dicts):
    return {"solar": len(dicts[0]), "won": len(dicts[1]), "wof": len(dicts[2]), "cons": len(dicts[3]), "prices_q15": len(dicts[4])}


def _ned_mlr_train_long_bg(s_day, e_day, batch_days):
    """
    Background task: haalt batches op, bouwt 1 grote trainingsset, fit OLS, schrijft coefs.
    """
    try:
        _publish_mqtt_discovery()
        _mqtt_publish_progress("running", {
            "start": s_day.isoformat(),
            "end": e_day.isoformat(),
            "batch_days": batch_days,
            "done": 0,
            "total": 0,
        })

        # NED discovery (1x)
        ids = find_ned_ids()
        if not ids:
            _mqtt_publish_progress("error", {"msg": "find_ned_ids faalde"})
            return

        tz_id    = ids["tz_id"]
        gran15   = ids["gran15"]
        act_prov = ids["act_prov"]
        act_cons = ids["act_cons"]
        cls_cur  = ids["cls_cur"] if ids["cls_cur"] is not None else ids["cls_back"]
        point_nl = POINT_NL if POINT_NL != 0 else ids["point_nl"]

        chunks = _date_chunks(s_day, e_day, batch_days)
        total = len(chunks)
        done = 0

        # Accumulators (zonder alles in geheugen te houden; we bouwen X/y per batch en voegen toe)
        X_all, y_all = [], []

        for (c_start, c_end) in chunks:
            cs, ce = c_start.isoformat(), c_end.isoformat()

            # NED kwartier: current/backcast (historische features)
            solar = fetch_ned_q15_range(cs, ce, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_cur)
            won   = fetch_ned_q15_range(cs, ce, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_cur)
            wof   = fetch_ned_q15_range(cs, ce, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_cur)
            cons  = {}
            if act_cons is not None:
                cons = fetch_ned_q15_range(cs, ce, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_cur)

            # ENTSO-E prijzen kwartier (Europe/Amsterdam), direct via REST
            prices_q15 = fetch_entsoe_prices_q15_range_api(cs, ce, eic_code=EIC_NL)

            # Normaliseer sleutels
            solar = _safe_canon(solar)
            won   = _safe_canon(won)
            wof   = _safe_canon(wof)
            cons  = _safe_canon(cons)
            prices_q15 = _safe_canon(prices_q15)

            # Indien geen verbruik: vul 0's op de unie van andere keys
            if not cons:
                all_ts = set(solar) | set(won) | set(wof)
                cons = {ts: 0.0 for ts in all_ts}

            # Append overlappende voorbeelden
            _append_xy(X_all, y_all, solar, won, wof, cons, prices_q15)

            done += 1
            # Voortgang updaten
            _mqtt_publish_progress(f"running:{done}/{total}", {
                "start": s_day.isoformat(),
                "end": e_day.isoformat(),
                "batch_days": batch_days,
                "done": done,
                "total": total,
                "last_chunk": {"start": cs, "end": ce},
                "last_sizes": _sum_len_dicts(solar, won, wof, cons, prices_q15),
                "n_samples_total": len(y_all),
            })
            # Zachte throttle tussen batches
            task.sleep(0.25)

        # Fitten op de samengestelde set
        if not X_all:
            log.error("ned_mlr_train_long: geen overlap features/prijzen over alle batches.")
            _mqtt_publish_progress("error", {
                "msg": "geen overlap features/prijzen",
                "n_samples": 0,
            })
            return

        coef = ols_fit(X_all, y_all, add_intercept=True)

        # Opslaan coefs met metadata
        meta_coef = {
            "features": ["solar","wind_on","wind_off","cons","lag_solar","lag_won","lag_wof","lag_cons"],
            "start": s_day.isoformat(),
            "end": e_day.isoformat(),
            "batch_days": batch_days,
            "n_chunks": total,
            "n_samples": len(y_all),
        }
        _save_coef(coef, meta)
        _mqtt_publish_coefficients(coef, meta_coef)

        _mqtt_publish_progress("done", {
            "start": s_day.isoformat(),
            "end": e_day.isoformat(),
            "batch_days": batch_days,
            "n_chunks": total,
            "n_samples": len(y_all),
        })
        log.info(f"ned_mlr_train_long: klaar ({total} batches, samples={len(y_all)})")

    except Exception as e:
        log.error(f"ned_mlr_train_long (bg): fout {e}")
        _mqtt_publish_progress("error", {"msg": str(e)})
        return


# -------------------------------
# PREDICT 7 dagen
# -------------------------------
@service
def ned_mlr_predict_7d(retain_days: int = 14):
    log.info("ned_mlr_predict_7d: start")

    # --- Load coefficients ---
    coef = _get_coef()
    if coef is None:
        log.error("Geen coef; run eerst pyscript.ned_mlr_train")
        return

    # --- Load RMSE per horizon from global backtest results ---
    global _backtest_results
    rmse_by_horizon = {}
    for h in range(1, 8):
        key = f"h{h}"
        rmse_by_horizon[h] = float(_backtest_results.get(key, {}).get("mean_rmse", 0.0))

    # --- Discovery ---
    ids = find_ned_ids()
    if not ids:
        return

    tz_id   = ids["tz_id"]
    gran15  = ids["gran15"]
    gran60  = ids["gran60"]
    act_prov= ids["act_prov"]
    act_cons= ids["act_cons"]
    cls_fore= ids["cls_fore"]
    cls_hist= ids["cls_cur"] if ids["cls_cur"] is not None else ids["cls_back"]
    point_nl= POINT_NL if POINT_NL != 0 else ids["point_nl"]

    # --- Time ---
    tz = ZoneInfo("Europe/Amsterdam")
    anchor_dt = datetime.now(tz=tz)
    anchor_day = anchor_dt.date()
    anchor_iso = anchor_day.isoformat()

    # --- Forecast range (7 days ahead) ---
    sdate = anchor_day.isoformat()
    edate = (anchor_day + timedelta(days=7)).isoformat()

    # --- Fetch forecast features ---
    solar15 = fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_fore)
    won15   = fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
    wof15   = fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
    cons15  = fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_fore) if act_cons else {}

    # fallback 60m
    if not solar15 and gran60:
        solar15 = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran60, tz_id, [point_nl], cls_fore))
    if not won15 and gran60:
        won15   = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore))
    if not wof15 and gran60:
        wof15   = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore))
    if not cons15 and gran60 and act_cons:
        cons15  = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran60, tz_id, [point_nl], cls_fore))

    if not solar15 or not won15 or not wof15:
        log.error("Forecast features onvolledig.")
        return

    if not cons15:
        cons15 = {}
        keys_union = set()
        for d in (solar15, won15, wof15):
            keys_union.update(d.keys())
        for k in keys_union:
            cons15[k] = 0.0

    # Normalize
    solar15 = _canonicalize_dict_keys(solar15)
    won15   = _canonicalize_dict_keys(won15)
    wof15   = _canonicalize_dict_keys(wof15)
    cons15  = _canonicalize_dict_keys(cons15)

    # --- Historical seed (yesterday) ---
    hist_start = anchor_dt - timedelta(days=1)
    h_sdate = hist_start.date().isoformat()
    h_edate = anchor_day.isoformat()

    hist_solar = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_hist))
    hist_won   = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist))
    hist_wof   = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist))
    hist_cons  = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_hist)) if act_cons else {}

    # --- Build feature matrix ---
    idx, Xf = build_feature_matrix_q15(solar15, won15, wof15, cons15)

    def _last(d):
        if not d:
            return 0.0
        ks = sorted(d.keys())
        return d[ks[-1]]

    seed = [
        _last(hist_solar),
        _last(hist_won),
        _last(hist_wof),
        _last(hist_cons) if hist_cons else 0.0
    ]

    for i in range(len(Xf)):
        if i == 0:
            for j in range(4, len(Xf[i])):
                Xf[i][j] = seed[j - 4]
        else:
            for j in range(4, len(Xf[i])):
                Xf[i][j] = Xf[i - 1][j - 4]

    # --- Predict (EUR/MWh) ---
    yhat_eur_mwh = ols_predict(Xf, coef, add_intercept=True)

    # --- Tax / toeslag ---
    nordpool_attrs = state.getattr("sensor.nordpool_electricity_prices") or {}
    tax = float(nordpool_attrs.get("tax", 1.21))
    additional_cost = float(nordpool_attrs.get("additional_cost", 0.13276))

    # --- Archive entries per horizon (center only) ---
    for i, ts in enumerate(idx):
        dt_start = datetime.fromisoformat(ts).astimezone(tz)
        dt_end   = dt_start + timedelta(minutes=15)

        pred_eur_mwh = float(yhat_eur_mwh[i])
        pred_date = dt_start.date()

        horizon = (pred_date - anchor_day).days + 1
        if horizon < 1 or horizon > 7:
            continue

        entry = {
            "start": dt_start.isoformat(),
            "end": dt_end.isoformat(),
            "center": pred_eur_mwh
        }

        day_key = pred_date.isoformat()
        archive_add_forecast(day_key, anchor_iso, horizon, [entry])

    archive_prune(retain_days)

    # --- Build flattened forecast for MQTT DIRECT uit voorspelling ---
    flat_entries = []
    last_price = None

    for i, ts in enumerate(idx):
        dt_start = datetime.fromisoformat(ts).astimezone(tz)
        pred_eur_mwh = float(yhat_eur_mwh[i])
        pred_date = dt_start.date()

        # horizon bepalen
        horizon = (pred_date - anchor_day).days + 1
        if horizon < 1 or horizon > 7:
            continue

        rmse = rmse_by_horizon.get(horizon, 0.0)

        center_mwh = pred_eur_mwh
        lower_mwh  = center_mwh - rmse
        upper_mwh  = center_mwh + rmse

        def conv(x):
            return round((x / 1000.0) * tax + additional_cost, 5)

        price_center = conv(center_mwh)
        price_lower  = conv(lower_mwh)
        price_upper  = conv(upper_mwh)

        # last_price direct bepalen
        if horizon == 1:
            last_price = price_center

        # alleen de benodigde velden opslaan
        flat_entries.append({
            "start": dt_start.isoformat(),
            "price_center": price_center,
            "price_lower": price_lower,
            "price_upper": price_upper,
        })

    # sorteer alles op tijd
    flat_entries.sort(key=lambda x: x["start"])

    payload = {
        "value": last_price if last_price is not None else 0,
        "data": flat_entries,
    }

    mqtt.publish(
        topic=MQTT_SENSOR_FORECAST_STATE_TOPIC,
        payload=json.dumps(payload),
        qos=1,
        retain=True,
    )

    log.info("ned_mlr_predict_7d: klaar")

# -------------------------------
# BACKTEST nieuw(per horizon D+1..D+H)
# -------------------------------
@service
def ned_mlr_backtest_forecasted(start_date: str = None, end_date: str = None):
    log.info("ned_mlr_backtest_forecasted: start")

    tz = ZoneInfo("Europe/Amsterdam")
    today = datetime.now(tz=tz).date()
    latest_hist = today - timedelta(days=1)

    global _forecast_archive_days
    days = _forecast_archive_days

    if not days:
        log.error("Forecast-archief leeg; run predict_7d eerst.")
        return

    # --- Verzamel alle doeldatums ---
    all_dates = []
    for d in days.keys():
        try:
            all_dates.append(datetime.fromisoformat(d).date())
        except:
            pass

    if not all_dates:
        log.error("Geen geldige datums in archief.")
        return

    all_dates.sort()

    # --- Bepaal analyse-range ---
    if start_date:
        try:
            s_day = datetime.fromisoformat(start_date).date()
        except:
            s_day = all_dates[0]
    else:
        s_day = all_dates[0]

    if end_date:
        try:
            e_day = datetime.fromisoformat(end_date).date()
        except:
            e_day = all_dates[-1]
    else:
        e_day = all_dates[-1]

    # Alleen dagen met echte prijzen
    if e_day > latest_hist:
        e_day = latest_hist

    if e_day < s_day:
        _mqtt_publish_backtest_forecasted("ok:0", {"msg": "geen historische dagen"})
        return

    # --- Aggregatie per horizon ---
    agg = {}
    for h in range(1, 8):
        agg[h] = {"mae": [], "rmse": [], "r2": [], "dates": []}

    # --- Loop over alle doeldatums ---
    cur = s_day
    while cur <= e_day:
        dkey = cur.isoformat()
        payload = days.get(dkey)

        if not payload or "forecasts" not in payload:
            cur += timedelta(days=1)
            continue

        # --- Echte prijzen ophalen ---
        true_day = fetch_entsoe_prices_q15_range_api(
            cur.isoformat(),
            (cur + timedelta(days=1)).isoformat(),
            eic_code=EIC_NL
        )
        true_day = _canonicalize_dict_keys(true_day)

        # --- Per horizon voorspellingen verzamelen ---
        preds_by_h = {h: [] for h in range(1, 8)}
        trues_by_h = {h: [] for h in range(1, 8)}

        for fc in payload["forecasts"]:
            h = fc.get("horizon")
            if h not in range(1, 8):
                continue

            for e in fc.get("data", []):
                ts = e.get("start")
                if ts not in true_day:
                    continue

                try:
                    pred = float(e["center"])
                    true = float(true_day[ts])
                except:
                    continue

                preds_by_h[h].append(pred)
                trues_by_h[h].append(true)

        # --- Metrics per horizon ---
        for h in range(1, 8):
            yp = preds_by_h[h]
            yt = trues_by_h[h]

            if yt and yp and len(yt) == len(yp):
                m = _metrics_no_gen(yt, yp)
                if m["mae"] is not None:
                    agg[h]["mae"].append(m["mae"])
                    agg[h]["rmse"].append(m["rmse"])
                if m["r2"] is not None:
                    agg[h]["r2"].append(m["r2"])
                agg[h]["dates"].append(dkey)

        cur += timedelta(days=1)

    # --- Samenvatting ---
    summary = {}
    counted = 0

    for h in range(1, 8):
        mae_list = agg[h]["mae"]
        rmse_list = agg[h]["rmse"]
        r2_list = agg[h]["r2"]

        summary[f"h{h}"] = {
            "mean_mae": sum(mae_list) / len(mae_list) if mae_list else None,
            "mean_rmse": sum(rmse_list) / len(rmse_list) if rmse_list else None,
            "mean_r2": sum(r2_list) / len(r2_list) if r2_list else None,
            "n_days": len(agg[h]["dates"]),
            "dates": agg[h]["dates"]
        }

        if agg[h]["dates"]:
            counted += 1

    global _backtest_results
    _backtest_results = summary
    _save_backtest_results()

    _mqtt_publish_backtest_forecasted(f"ok:{counted}", summary)
    log.info("ned_mlr_backtest_forecasted: klaar")


# -------------------------------
# Bootstrap archive (archief initiel vullen)
# -------------------------------
@service
def ned_mlr_bootstrap_archive(days: int = 14):
    """
    Bouwt het archief alsof predict_7d de afgelopen 'days' dagen al gedraaid had.
    Elke anchor-dag levert 7 horizons op (H1..H7).
    Bij ontbrekende forecast-features worden lege horizons toegevoegd,
    zodat de archiefstructuur altijd consistent blijft.
    """

    log.info(f"ned_mlr_bootstrap_archive: start voor {days} dagen")

    # --- Load coefficients ---
    coef = _get_coef()
    if coef is None:
        log.error("Geen coef; run eerst pyscript.ned_mlr_train")
        return

    # --- Discovery ---
    ids = find_ned_ids()
    if not ids:
        return

    tz_id   = ids["tz_id"]
    gran15  = ids["gran15"]
    gran60  = ids["gran60"]
    act_prov= ids["act_prov"]
    act_cons= ids["act_cons"]
    cls_fore= ids["cls_fore"]
    cls_hist= ids["cls_cur"] if ids["cls_cur"] is not None else ids["cls_back"]
    point_nl= POINT_NL if POINT_NL != 0 else ids["point_nl"]

    tz = ZoneInfo("Europe/Amsterdam")
    today = datetime.now(tz=tz).date()

    # --- Loop over anchors in het verleden ---
    for offset in range(days, 0, -1):
        anchor_day = today - timedelta(days=offset)
        anchor_iso = anchor_day.isoformat()
        log.info(f"Bootstrap anchor {anchor_iso}")

        # --- Forecast range ---
        sdate = anchor_day.isoformat()
        edate = (anchor_day + timedelta(days=7)).isoformat()

        # --- Fetch forecast features ---
        solar15 = fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_fore)
        won15   = fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
        wof15   = fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_fore)
        cons15  = fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_fore) if act_cons else {}

        # --- Fallback naar 60m ---
        if not solar15 and gran60:
            solar15 = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, SOLAR_TYPE_ID, act_prov, gran60, tz_id, [point_nl], cls_fore))
        if not won15 and gran60:
            won15   = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, WIND_ON_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore))
        if not wof15 and gran60:
            wof15   = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, WIND_OFF_TYPE_ID, act_prov, gran60, tz_id, [point_nl, 36, 14], cls_fore))
        if not cons15 and gran60 and act_cons:
            cons15  = _upsample_hour_dict_to_q15_local(fetch_ned_q15_range(sdate, edate, CONSUMPTION_TYPE_ID, act_cons, gran60, tz_id, [point_nl], cls_fore))

        # --- Als features ontbreken → lege horizons toevoegen ---
        if not solar15 or not won15 or not wof15:
            log.warning(f"Anchor {anchor_iso}: onvolledige features, maar dag wordt wel aangemaakt")

            for h in range(1, 8):
                pred_date = anchor_day + timedelta(days=h-1)
                day_key = pred_date.isoformat()
                archive_add_forecast(day_key, anchor_iso, h, [])
            continue

        # --- Normalize ---
        solar15 = _canonicalize_dict_keys(solar15)
        won15   = _canonicalize_dict_keys(won15)
        wof15   = _canonicalize_dict_keys(wof15)
        cons15  = _canonicalize_dict_keys(cons15)

        # --- Historical seed (yesterday) ---
        hist_start = anchor_day - timedelta(days=1)
        h_sdate = hist_start.isoformat()
        h_edate = anchor_day.isoformat()

        hist_solar = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, SOLAR_TYPE_ID, act_prov, gran15, tz_id, [point_nl], cls_hist))
        hist_won   = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, WIND_ON_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist))
        hist_wof   = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, WIND_OFF_TYPE_ID, act_prov, gran15, tz_id, [point_nl, 36, 14], cls_hist))
        hist_cons  = _canonicalize_dict_keys(fetch_ned_q15_range(h_sdate, h_edate, CONSUMPTION_TYPE_ID, act_cons, gran15, tz_id, [point_nl], cls_hist)) if act_cons else {}

        # --- Feature matrix ---
        idx, Xf = build_feature_matrix_q15(solar15, won15, wof15, cons15)

        def _last(d):
            if not d:
                return 0.0
            ks = sorted(d.keys())
            return d[ks[-1]]

        seed = [
            _last(hist_solar),
            _last(hist_won),
            _last(hist_wof),
            _last(hist_cons) if hist_cons else 0.0
        ]

        for i in range(len(Xf)):
            if i == 0:
                for j in range(4, len(Xf[i])):
                    Xf[i][j] = seed[j - 4]
            else:
                for j in range(4, len(Xf[i])):
                    Xf[i][j] = Xf[i - 1][j - 4]

        # --- Predict ---
        yhat_eur_mwh = ols_predict(Xf, coef, add_intercept=True)

        # --- Archive per horizon ---
        for i, ts in enumerate(idx):
            dt_start = datetime.fromisoformat(ts)
            dt_end   = dt_start + timedelta(minutes=15)

            pred_date = dt_start.date()
            horizon = (pred_date - anchor_day).days + 1
            if horizon < 1 or horizon > 7:
                continue

            entry = {
                "start": dt_start.isoformat(),
                "end": dt_end.isoformat(),
                "center": float(yhat_eur_mwh[i])
            }

            day_key = pred_date.isoformat()
            archive_add_forecast(day_key, anchor_iso, horizon, [entry])

    archive_prune(days)
    log.info("ned_mlr_bootstrap_archive: klaar")

# -------------------------------
# archief leeggooien
# -------------------------------
@service
def ned_mlr_reset_forecast_archive():
    global _forecast_archive_days
    _forecast_archive_days = {}
    _save_forecast_archive()
    log.warning("ned_mlr_reset_forecast_archive: archive leeggemaakt")
