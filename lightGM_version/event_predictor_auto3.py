import os
import csv
import time
import glob
import bisect
import re
import warnings
import hashlib
import json
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 机器学习库
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

# Bestdori API
from bestdori.events import Event
from bestdori.eventtracker import EventTracker

warnings.filterwarnings('ignore')

# ==========================================
#              全局配置管理系统
# ==========================================
DEFAULT_CONFIG = {
    # 基础配置
    "TARGET_EVENT_ID": 295,
    "HISTORY_CHECK_RANGE": 100,
    "MIN_HISTORY_EVENT_ID": 226,
    "PREDICTION_STEP_MINUTES": 30,

    # 核心模型参数
    "WINDOW_SIZE": 4,  # 滞后特征窗口大小
    "OVERLAP_HOURS": 2,  # 切片重叠时间
    "END_DURATION_HOURS": 22,  # 冲刺阶段时长
    "END_REFERENCE_HOURS": 24,  # 冲刺参考时长
    "PENULTIMATE_DURATION_HOURS": 24,
    "PENULTIMATE_REFERENCE_HOURS": 24,

    # 损失函数权重 (非对称)
    "ASYM_UNDER_WEIGHT": 1.1,
    "ASYM_OVER_WEIGHT": 1.0,
    "PENULTIMATE_ASYM_UNDER_WEIGHT": 6.0,
    "PENULTIMATE_ASYM_OVER_WEIGHT": 0.7,
    "END_ASYM_UNDER_WEIGHT": 8.0,
    "END_ASYM_OVER_WEIGHT": 0.5,

    # 约束条件
    "MAX_HOURLY_SPEED": 700000
}

CONFIG_FILE = 'config.json'
CONFIG = DEFAULT_CONFIG.copy()


def load_config():
    """加载配置，如果文件存在则覆盖默认值"""
    global CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                CONFIG.update(loaded)
        except Exception as e:
            print(f"[Config] Load failed: {e}")
    return CONFIG


def save_config(new_config):
    """保存配置到文件"""
    global CONFIG
    try:
        # 只更新有效的键，并进行类型转换
        for k in DEFAULT_CONFIG.keys():
            if k in new_config:
                default_val = DEFAULT_CONFIG[k]
                if isinstance(default_val, int):
                    CONFIG[k] = int(new_config[k])
                elif isinstance(default_val, float):
                    CONFIG[k] = float(new_config[k])
                else:
                    CONFIG[k] = new_config[k]

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=4)
        return True
    except Exception as e:
        print(f"[Config] Save failed: {e}")
        return False


# 初始化加载配置
load_config()


def get_config(key):
    return CONFIG.get(key, DEFAULT_CONFIG.get(key))


def get_ms_constants():
    """获取基于配置的毫秒级常量"""
    return {
        "PREDICTION_STEP_MS": get_config("PREDICTION_STEP_MINUTES") * 60 * 1000,
        "OVERLAP_MS": get_config("OVERLAP_HOURS") * 3600 * 1000,
        "MS_PER_HOUR": 3600 * 1000,
        "MS_PER_DAY": 24 * 3600 * 1000
    }


# 路径常量
EVENT_DATA_DIR = 'event_data'
MODEL_DIR = 'models_overlap'
TEMP_CURRENT_FILE = 'current_event_temp.csv'
CACHE_FILE = 'event_types_cache.json'
LATEST_EVENT_CACHE_FILE = 'latest_event_cache.json'
CURRENT_EVENT_INFO_FILE = 'current_event_info.json'  # 新增：用于记录当前临时文件所属的活动ID


# ==========================================
#           第一部分：基础工具
# ==========================================

def get_latest_event_id(server=3):
    now_ts_ms = int(time.time() * 1000)
    # 优先查缓存
    if os.path.exists(LATEST_EVENT_CACHE_FILE):
        try:
            with open(LATEST_EVENT_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if 'id' in cache and 'end_at' in cache:
                    if now_ts_ms < int(cache['end_at']):
                        return cache['id']
        except:
            pass

    print("正在自动检测最新活动(联网)...")
    url = "https://bestdori.com/api/events/all.5.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200: return None
        data = resp.json()
        latest_event_id = None
        max_start_time = -1
        latest_end_at = 0
        latest_start_at = 0

        for eid_str, info in data.items():
            if 'startAt' not in info or len(info['startAt']) <= server or info['startAt'][server] is None: continue
            start_at = float(info['startAt'][server])

            if start_at <= now_ts_ms:
                if start_at > max_start_time:
                    max_start_time = start_at
                    latest_event_id = int(eid_str)
                    latest_start_at = start_at
                    latest_end_at = float(info['endAt'][server]) if 'endAt' in info and info['endAt'][
                        server] else now_ts_ms + 86400000

        if latest_event_id:
            with open(LATEST_EVENT_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({'id': latest_event_id, 'start_at': latest_start_at, 'end_at': latest_end_at}, f)
        return latest_event_id
    except:
        return None


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def get_event_metadata(event_id):
    try:
        e = Event(event_id)
        info = e.get_info()
        if not info: return None
        event_type = info.get('eventType', 'unknown')
        event_names = info.get('eventName')
        if isinstance(event_names, list):
            name = event_names[3] if len(event_names) > 3 and event_names[3] else (
                event_names[0] if len(event_names) > 0 else f"Event {event_id}")
        else:
            name = f"Event {event_id}"

        start_ts = info['startAt'][3]
        end_ts = info['endAt'][3]
        start_time_dt = datetime.utcfromtimestamp(int(start_ts) / 1000) + timedelta(hours=8)
        end_time_dt = datetime.utcfromtimestamp(int(end_ts) / 1000) + timedelta(hours=8)

        return {
            'id': event_id, 'type': event_type, 'name': name,
            'start_ts': float(start_ts), 'end_ts': float(end_ts),
            'start_time': start_time_dt,
            'end_time': end_time_dt,
            'end_time_str': end_time_dt.strftime("%Y-%m-%d %H:%M:%S")
        }
    except:
        return None


def save_single_event_data(event_id, event_type, server=3):
    save_dir = os.path.join(EVENT_DATA_DIR, event_type)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"event_{event_id}.csv")
    if os.path.exists(filename): return True
    try:
        tracker = EventTracker(event=event_id, server=server)
        data = tracker.get_data(1000)
        if not data.get('result') or not data.get('cutoffs'): return False
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time_point(ms)', 'time(UTC+8)', 'pt(ep)'])
            for point in data['cutoffs']:
                dt = datetime.utcfromtimestamp(point['time'] / 1000) + timedelta(hours=8)
                writer.writerow([point['time'], dt.strftime("%Y-%m-%d %H:%M:%S"), point['ep']])
        time.sleep(0.5)
        return True
    except:
        return False


def ensure_historical_data(target_type, current_id, progress_callback=None):
    if progress_callback: progress_callback(0, "正在检查历史数据...")
    cache = load_cache()
    stop_id = max(get_config("MIN_HISTORY_EVENT_ID") - 1, current_id - get_config("HISTORY_CHECK_RANGE"))
    check_list = range(current_id - 1, stop_id, -1)

    for i, eid in enumerate(check_list):
        if progress_callback and i % 5 == 0: progress_callback(int((i / len(check_list)) * 20), f"扫描历史 {eid}...")
        str_eid = str(eid)
        
        if str_eid not in cache:
            try:
                info = Event(eid).get_info()
                if info:
                    cache[str_eid] = info.get('eventType', 'unknown')
                    save_cache(cache)
            except:
                pass
                
        if cache.get(str_eid) == target_type:
            save_single_event_data(eid, target_type)


def fetch_current_event_data(event_id, server=3):
    """
    带重试(Retry)和合并(Merge)逻辑的数据获取函数。
    修改：增加活动ID检查，切换活动时自动清理旧的临时文件。
    """
    # === 检查活动ID是否变更 ===
    should_clean = False
    if os.path.exists(CURRENT_EVENT_INFO_FILE):
        try:
            with open(CURRENT_EVENT_INFO_FILE, 'r') as f:
                info = json.load(f)
                if info.get('event_id') != event_id:
                    should_clean = True
        except:
            should_clean = True
    else:
        # 如果信息文件不存在，但数据文件存在，说明可能是旧代码遗留或异常情况，安全起见清理
        if os.path.exists(TEMP_CURRENT_FILE):
            should_clean = True

    if should_clean:
        print(f"[Core] 检测到活动切换或缓存失效 (Target: {event_id})，清理旧数据文件...")
        try:
            if os.path.exists(TEMP_CURRENT_FILE):
                os.remove(TEMP_CURRENT_FILE)
        except Exception as e:
            print(f"[Core] 清理失败: {e}")

    # 更新当前活动信息
    try:
        with open(CURRENT_EVENT_INFO_FILE, 'w') as f:
            json.dump({'event_id': event_id, 'updated_at': time.time()}, f)
    except Exception as e:
        print(f"[Core] 保存活动信息失败: {e}")
    # ========================

    print(f"[Core] 正在尝试获取 Event {event_id} 数据...")
    max_retries = 5
    retry_delay = 2

    new_data_points = []
    success = False

    # 1. 重试循环 (Retry Loop)
    for attempt in range(max_retries):
        try:
            tracker = EventTracker(event=event_id, server=server)
            data = tracker.get_data(1000)  # 获取最近的点

            if data and 'cutoffs' in data:
                cutoffs = data['cutoffs']
                if cutoffs:
                    new_data_points = cutoffs
                    success = True
                    break
        except Exception as e:
            print(f"[Core] Fetch attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    if not success:
        print("[Core] 多次重试后无法获取新数据。")
        return False

    # 2. 读取现有本地数据进行合并 (Merge)
    existing_df = pd.DataFrame()
    if os.path.exists(TEMP_CURRENT_FILE):
        try:
            existing_df = pd.read_csv(TEMP_CURRENT_FILE)
        except Exception as e:
            print(f"[Core] Warning: 现有文件损坏或不可读, 将覆盖。 {e}")

    # 将新数据转为 DataFrame
    new_rows = []
    for point in new_data_points:
        dt = datetime.utcfromtimestamp(point['time'] / 1000) + timedelta(hours=8)
        new_rows.append({
            'time_point(ms)': point['time'],
            'time(UTC+8)': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'pt(ep)': point['ep']
        })
    new_df = pd.DataFrame(new_rows)

    # 合并与去重 (Deduplicate)
    if not existing_df.empty:
        # 合并
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # 根据时间戳去重，保留最新的
        combined_df.drop_duplicates(subset=['time_point(ms)'], keep='last', inplace=True)
        # 排序
        combined_df.sort_values(by='time_point(ms)', inplace=True)
    else:
        combined_df = new_df
        combined_df.sort_values(by='time_point(ms)', inplace=True)

    # 3. 保存回文件
    try:
        combined_df.to_csv(TEMP_CURRENT_FILE, index=False, encoding='utf-8')
        print(f"[Core] 数据已合并并保存，总点数: {len(combined_df)}")
        return True
    except Exception as e:
        print(f"[Core] 保存合并数据失败: {e}")
        return False


# ==========================================
#           第二部分：增强型切片逻辑 (Overlap)
# ==========================================

def calculate_simple_day_progress(df):
    consts = get_ms_constants()
    normalized = df['time(UTC+8)'].dt.normalize()
    seconds = (df['time(UTC+8)'] - normalized).dt.total_seconds()
    return seconds * 1000 / consts['MS_PER_DAY']


def get_pt_at_ms_generic(target_ms, _df):
    """辅助函数：根据 elapsed_ms 插值获取 PT"""
    if target_ms < _df['elapsed_ms'].min(): return _df['pt(ep)'].iloc[0]
    if target_ms > _df['elapsed_ms'].max(): return _df['pt(ep)'].iloc[-1]

    idx = bisect.bisect_left(_df['elapsed_ms'].values, target_ms)
    if idx == 0: return _df['pt(ep)'].iloc[0]
    if idx >= len(_df): return _df['pt(ep)'].iloc[-1]

    p1 = _df.iloc[idx - 1]
    p2 = _df.iloc[idx]
    t1, pt1 = p1['elapsed_ms'], p1['pt(ep)']
    t2, pt2 = p2['elapsed_ms'], p2['pt(ep)']

    if t2 == t1: return pt1
    return pt1 + (pt2 - pt1) * (target_ms - t1) / (t2 - t1)


def prepare_slices(directory, target_event_type, target_event_id):
    consts = get_ms_constants()
    files = glob.glob(os.path.join(directory, target_event_type, 'event_*.csv'))
    slices = {'start': [], 'middle': [], 'penultimate': [], 'end': []}

    print(f"正在进行切片处理... (使用配置 Window={get_config('WINDOW_SIZE')})")

    for f in files:
        eid = int(re.search(r'event_(\d+)', f).group(1))
        if eid == target_event_id or eid < get_config("MIN_HISTORY_EVENT_ID"): continue

        try:
            df = pd.read_csv(f)
            df['time(UTC+8)'] = pd.to_datetime(df['time(UTC+8)'])
            df['elapsed_ms'] = df['time_point(ms)'] - df['time_point(ms)'].iloc[0]

            total_duration_hours = df['elapsed_ms'].iloc[-1] / consts['MS_PER_HOUR']
            total_final_pt = df['pt(ep)'].iloc[-1]
            event_intensity = total_final_pt / total_duration_hours if total_duration_hours > 0 else 0
            df['event_intensity'] = event_intensity

            df['pt_diff'] = df['pt(ep)'].diff().fillna(0)
            df['time_diff'] = df['elapsed_ms'].diff().fillna(consts['PREDICTION_STEP_MS'])
            df.loc[df['time_diff'] <= 0, 'time_diff'] = consts['PREDICTION_STEP_MS']
            df['pt_rate'] = df['pt_diff'] / df['time_diff']

            # 24h lookup
            df_lookup = df[['time(UTC+8)', 'pt_rate']].copy()
            df_lookup['lookup_time'] = df_lookup['time(UTC+8)'] + timedelta(hours=24)
            df = df.sort_values('time(UTC+8)')
            df_lookup = df_lookup.sort_values('lookup_time')
            df = pd.merge_asof(
                df, df_lookup[['lookup_time', 'pt_rate']],
                left_on='time(UTC+8)', right_on='lookup_time',
                direction='nearest', tolerance=pd.Timedelta(minutes=45), suffixes=('', '_24h')
            )
            df['pt_rate_24h'] = df['pt_rate_24h'].fillna(0) if 'pt_rate_24h' in df.columns else 0
            if 'lookup_time' in df.columns: df = df.drop(columns=['lookup_time'])

            start_time = df['time(UTC+8)'].iloc[0]
            total_ms = df['elapsed_ms'].iloc[-1]
            df['time_until_end_ms'] = total_ms - df['elapsed_ms']

            # Phases
            end_phase_dur_ms = get_config("END_DURATION_HOURS") * 3600 * 1000
            penultimate_dur_ms = get_config("PENULTIMATE_DURATION_HOURS") * 3600 * 1000
            end_start_ms = total_ms - end_phase_dur_ms
            penultimate_start_ms = end_start_ms - penultimate_dur_ms
            middle_end_limit_ms = penultimate_start_ms + (2 * 3600 * 1000)

            # Start
            next_midnight = (start_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            start_slice_end = next_midnight + timedelta(hours=get_config("OVERLAP_HOURS"))
            start_df = df[df['time(UTC+8)'] < start_slice_end].copy()
            if not start_df.empty:
                start_df['is_weekend'] = start_df['time(UTC+8)'].dt.weekday >= 5
                start_df['day_progress'] = calculate_simple_day_progress(start_df)
                slices['start'].append(start_df)

            # Middle
            curr_slice_start = next_midnight
            while True:
                curr_slice_start_ms = (curr_slice_start - start_time).total_seconds() * 1000
                if curr_slice_start_ms >= middle_end_limit_ms: break
                curr_slice_end = curr_slice_start + timedelta(days=1, hours=get_config("OVERLAP_HOURS"))
                mid_df = df[(df['time(UTC+8)'] >= curr_slice_start) & (df['time(UTC+8)'] < curr_slice_end)].copy()
                if not mid_df.empty:
                    mid_end_ms = mid_df['elapsed_ms'].max()
                    if mid_end_ms <= penultimate_start_ms + (6 * 3600 * 1000) and len(mid_df) > get_config(
                            "WINDOW_SIZE") + 2:
                        mid_df['day_ms'] = (mid_df['time(UTC+8)'] - curr_slice_start).dt.total_seconds() * 1000
                        mid_df['day_progress'] = mid_df['day_ms'] / consts['MS_PER_DAY']
                        mid_df['is_weekend'] = (curr_slice_start.weekday() >= 5)
                        slices['middle'].append(mid_df)
                curr_slice_start += timedelta(days=1)
                if curr_slice_start_ms > total_ms: break

            # Penultimate
            ante_ref_dur_ms = get_config("PENULTIMATE_REFERENCE_HOURS") * 3600 * 1000
            ante_start_ms = penultimate_start_ms - ante_ref_dur_ms
            pen_df = df[(df['elapsed_ms'] >= penultimate_start_ms) & (df['elapsed_ms'] < end_start_ms)].copy()
            if not pen_df.empty and ante_start_ms > 0:
                pt_pen_start = get_pt_at_ms_generic(penultimate_start_ms, df)
                pt_ante_start = get_pt_at_ms_generic(ante_start_ms, df)
                ante_growth = pt_pen_start - pt_ante_start
                if ante_growth > 0:
                    pen_df['base_growth'] = ante_growth
                    pen_df['is_weekend'] = pen_df['time(UTC+8)'].dt.weekday >= 5
                    pen_df['day_progress'] = calculate_simple_day_progress(pen_df)
                    slices['penultimate'].append(pen_df)

            # End
            end_ref_dur_ms = get_config("END_REFERENCE_HOURS") * 3600 * 1000
            end_ref_start_ms = end_start_ms - end_ref_dur_ms
            end_df = df[df['elapsed_ms'] >= end_start_ms].copy()
            if not end_df.empty:
                pt_end_start = get_pt_at_ms_generic(end_start_ms, df)
                pt_ref_start = get_pt_at_ms_generic(end_ref_start_ms, df)
                pen_growth = pt_end_start - pt_ref_start
                if pen_growth > 0:
                    end_df['base_growth'] = pen_growth
                    end_df['is_weekend'] = end_df['time(UTC+8)'].dt.weekday >= 5
                    end_df['day_progress'] = calculate_simple_day_progress(end_df)
                    slices['end'].append(end_df)
        except Exception:
            continue
    return slices


# ==========================================
#           第三部分：模型训练
# ==========================================

def make_lags(df, lags, target_col='pt_rate_norm'):
    df = df.copy()
    for l in range(1, lags + 1): df[f'lag_{l}'] = df[target_col].shift(l)
    return df.dropna()


def objective_normal(y_true, y_pred):
    res = (y_true - y_pred).astype("float")
    grad = np.where(res > 0, -2 * get_config("ASYM_UNDER_WEIGHT") * res, -2 * get_config("ASYM_OVER_WEIGHT") * res)
    hess = np.where(res > 0, 2 * get_config("ASYM_UNDER_WEIGHT"), 2 * get_config("ASYM_OVER_WEIGHT"))
    return grad, hess


def objective_penultimate(y_true, y_pred):
    res = (y_true - y_pred).astype("float")
    grad = np.where(res > 0, -2 * get_config("PENULTIMATE_ASYM_UNDER_WEIGHT") * res,
                    -2 * get_config("PENULTIMATE_ASYM_OVER_WEIGHT") * res)
    hess = np.where(res > 0, 2 * get_config("PENULTIMATE_ASYM_UNDER_WEIGHT"),
                    2 * get_config("PENULTIMATE_ASYM_OVER_WEIGHT"))
    return grad, hess


def objective_aggressive(y_true, y_pred):
    res = (y_true - y_pred).astype("float")
    grad = np.where(res > 0, -2 * get_config("END_ASYM_UNDER_WEIGHT") * res,
                    -2 * get_config("END_ASYM_OVER_WEIGHT") * res)
    hess = np.where(res > 0, 2 * get_config("END_ASYM_UNDER_WEIGHT"), 2 * get_config("END_ASYM_OVER_WEIGHT"))
    return grad, hess


def train_segment_model(df_list, segment_type, scalers_ref):
    if not df_list: return None
    training_data = []

    # 选择损失函数
    if segment_type == 'end':
        loss_func = objective_aggressive
    elif segment_type == 'penultimate':
        loss_func = objective_penultimate
    else:
        loss_func = objective_normal

    for df in df_list:
        sub_df = df.copy()
        # 归一化特征
        sub_df['pt_rate_norm'] = scalers_ref['pt_rate'].transform(sub_df['pt_rate'].values.reshape(-1, 1)).ravel()
        sub_df['elapsed_ms_norm'] = scalers_ref['elapsed_ms'].transform(
            sub_df['elapsed_ms'].values.reshape(-1, 1)).ravel()
        sub_df['time_until_end_ms_norm'] = scalers_ref['time_until_end_ms'].transform(
            sub_df['time_until_end_ms'].values.reshape(-1, 1)).ravel()
        sub_df['event_intensity_norm'] = scalers_ref['event_intensity'].transform(
            sub_df['event_intensity'].values.reshape(-1, 1)).ravel()

        if segment_type in ['end', 'penultimate']:
            ref_hours = get_config("END_REFERENCE_HOURS") if segment_type == 'end' else get_config(
                "PENULTIMATE_REFERENCE_HOURS")
            base_growth = sub_df['base_growth'].iloc[0]
            avg_hourly_growth = max(base_growth / ref_hours, 1)

            sub_df['target_scaled_rate'] = sub_df['pt_rate'] / avg_hourly_growth
            raw_inv = 1.0 / (sub_df['time_until_end_ms'] / 3600000.0 + 0.25)
            sub_df['inv_time_hr_norm'] = scalers_ref['inv_time_hr'].transform(raw_inv.values.reshape(-1, 1)).ravel()
        elif segment_type == 'start':
            sub_df = make_lags(sub_df, get_config("WINDOW_SIZE"), 'pt_rate_norm')
        else:
            sub_df['pt_rate_24h_norm'] = scalers_ref['pt_rate'].transform(
                sub_df['pt_rate_24h'].values.reshape(-1, 1)).ravel()
            sub_df['log_rate'] = np.log1p(sub_df['pt_rate'])
            sub_df['log_rate_24h'] = np.log1p(sub_df['pt_rate_24h'])
            sub_df['target_log_diff'] = sub_df['log_rate'] - sub_df['log_rate_24h']
        training_data.append(sub_df)

    full_df = pd.concat(training_data, ignore_index=True)
    full_df['hour_sin'] = np.sin(2 * np.pi * full_df['day_progress'])
    full_df['hour_cos'] = np.cos(2 * np.pi * full_df['day_progress'])

    sample_weights = None
    if segment_type == 'end':
        hours_until_end = full_df['time_until_end_ms'] / 3600000.0
        sample_weights = (1.0 + 8.0 / (hours_until_end + 0.5)).values

    features = []
    if segment_type in ['end', 'penultimate']:
        features = ['is_weekend', 'day_progress', 'hour_sin', 'hour_cos', 'time_until_end_ms_norm', 'inv_time_hr_norm',
                    'event_intensity_norm']
        target_col = 'target_scaled_rate'
    elif segment_type == 'start':
        features = [f'lag_{i}' for i in range(1, get_config("WINDOW_SIZE") + 1)]
        features += ['is_weekend', 'day_progress', 'hour_sin', 'hour_cos', 'elapsed_ms_norm']
        target_col = 'pt_rate_norm'
    else:
        features = ['is_weekend', 'day_progress', 'hour_sin', 'hour_cos', 'event_intensity_norm']
        target_col = 'target_log_diff'

    X = full_df[features]
    y = full_df[target_col]

    valid_idx = ~X.isnull().any(axis=1) & ~y.isnull()
    X = X[valid_idx]
    y = y[valid_idx]
    if sample_weights is not None: sample_weights = sample_weights[valid_idx]

    model = LGBMRegressor(
        n_estimators=1500, learning_rate=0.01, num_leaves=62, min_child_samples=10,
        random_state=42, n_jobs=-1, objective=loss_func
    )
    model.fit(X, y, sample_weight=sample_weights)
    return model, features


def build_segmented_models(directory, target_event_type, target_event_id, progress_callback=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 哈希包含配置值，配置改变触发重训
    hasher = hashlib.md5()
    config_state = f"{target_event_type}_v3_{get_config('WINDOW_SIZE')}_{get_config('ASYM_UNDER_WEIGHT')}"
    hasher.update(config_state.encode('utf-8'))
    model_hash = hasher.hexdigest()
    path_prefix = os.path.join(MODEL_DIR, f"{model_hash}")

    if os.path.exists(f"{path_prefix}_start.pkl"):
        if progress_callback: progress_callback(70, "加载缓存的分段模型...")
        return {
            'start': joblib.load(f"{path_prefix}_start.pkl"),
            'middle': joblib.load(f"{path_prefix}_middle.pkl"),
            'penultimate': joblib.load(f"{path_prefix}_penultimate.pkl"),
            'end': joblib.load(f"{path_prefix}_end.pkl"),
            'scalers': joblib.load(f"{path_prefix}_scalers.pkl"),
            'features': joblib.load(f"{path_prefix}_features.pkl")
        }

    if progress_callback: progress_callback(30, "正在切片与预处理 (配置已变更)...")
    slices = prepare_slices(directory, target_event_type, target_event_id)
    if not any(slices.values()): raise ValueError("没有足够数据训练")

    all_dfs = slices['start'] + slices['middle'] + slices['penultimate'] + slices['end']
    big_df = pd.concat(all_dfs, ignore_index=True)

    # 训练 Scalers
    scaler_rate = MinMaxScaler(feature_range=(0, 10))
    scaler_rate.fit(big_df[['pt_rate']])
    scaler_elapsed = MinMaxScaler(feature_range=(0, 1))
    scaler_elapsed.fit(big_df[['elapsed_ms']])
    scaler_end = MinMaxScaler(feature_range=(0, 1))
    scaler_end.fit(big_df[['time_until_end_ms']])

    big_df['inv_time_hr'] = 1.0 / (big_df['time_until_end_ms'] / 3600000.0 + 0.25)
    scaler_inv = MinMaxScaler(feature_range=(0, 1))
    scaler_inv.fit(big_df[['inv_time_hr']])

    scaler_intensity = MinMaxScaler(feature_range=(0, 1))
    scaler_intensity.fit(big_df[['event_intensity']])

    scalers = {
        'pt_rate': scaler_rate,
        'elapsed_ms': scaler_elapsed,
        'time_until_end_ms': scaler_end,
        'inv_time_hr': scaler_inv,
        'event_intensity': scaler_intensity
    }

    models, feature_names = {}, {}
    phases = ['start', 'middle', 'penultimate', 'end']

    for idx, phase in enumerate(phases):
        if progress_callback: progress_callback(40 + idx * 10, f"训练 {phase} 模型...")
        data = slices[phase] if slices[phase] else slices['middle']  # 降级处理
        m, f = train_segment_model(data, phase, scalers)
        models[phase], feature_names[phase] = m, f
        joblib.dump(m, f"{path_prefix}_{phase}.pkl")

    joblib.dump(scalers, f"{path_prefix}_scalers.pkl")
    joblib.dump(feature_names, f"{path_prefix}_features.pkl")
    models['scalers'] = scalers
    models['features'] = feature_names
    return models


# ==========================================
#           第四部分：接力预测 (Buffer Relay)
# ==========================================

def get_24h_ago_raw_value(buffer, current_elapsed):
    target_elapsed = current_elapsed - 86400000
    if target_elapsed < buffer[0]['elapsed_ms']: return 0.0

    keys = [item['elapsed_ms'] for item in buffer]
    idx = bisect.bisect_left(keys, target_elapsed)
    if idx >= len(buffer): return buffer[-1]['pt_rate_raw']

    item_after = buffer[idx]
    if idx > 0:
        item_before = buffer[idx - 1]
        if abs(item_after['elapsed_ms'] - target_elapsed) < abs(item_before['elapsed_ms'] - target_elapsed):
            return item_before['pt_rate_raw']
        else:
            return item_after['pt_rate_raw']
    return item_after['pt_rate_raw']


def get_pt_at_elapsed(buffer, target_elapsed):
    if target_elapsed < buffer[0]['elapsed_ms']: return buffer[0]['pt(ep)']
    if target_elapsed > buffer[-1]['elapsed_ms']: return buffer[-1]['pt(ep)']

    keys = [item['elapsed_ms'] for item in buffer]
    idx = bisect.bisect_left(keys, target_elapsed)
    if idx >= len(buffer): return buffer[-1]['pt(ep)']

    p2 = buffer[idx]
    if idx == 0: return p2['pt(ep)']
    p1 = buffer[idx - 1]

    t1, pt1 = p1['elapsed_ms'], p1['pt(ep)']
    t2, pt2 = p2['elapsed_ms'], p2['pt(ep)']
    if t2 == t1: return pt1
    return pt1 + (pt2 - pt1) * (target_elapsed - t1) / (t2 - t1)


def predict_relay(models, df_curr, meta, progress_callback=None):
    consts = get_ms_constants()
    if progress_callback: progress_callback(80, "开始接力预测...")

    start_time, end_time = meta['start_time'], meta['end_time']
    total_duration_ms = (end_time - start_time).total_seconds() * 1000
    scalers = models['scalers']

    df_curr['pt_rate_norm'] = scalers['pt_rate'].transform(df_curr['pt_rate'].values.reshape(-1, 1)).ravel()
    buffer = df_curr[['elapsed_ms', 'pt(ep)', 'pt_rate_norm', 'pt_rate']].rename(
        columns={'pt_rate': 'pt_rate_raw'}).to_dict('records')

    curr_elapsed, curr_pt = buffer[-1]['elapsed_ms'], buffer[-1]['pt(ep)']
    curr_hours = curr_elapsed / consts['MS_PER_HOUR']
    current_intensity_raw = curr_pt / curr_hours if curr_hours > 0 else 0
    current_intensity_norm = scalers['event_intensity'].transform([[current_intensity_raw]])[0][0]

    pred_pts, pred_times = [], []
    sim_elapsed = curr_elapsed

    end_phase_start_ms = total_duration_ms - (get_config("END_DURATION_HOURS") * 3600 * 1000)
    penultimate_phase_start_ms = end_phase_start_ms - (get_config("PENULTIMATE_DURATION_HOURS") * 3600 * 1000)
    first_midnight = (start_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    first_midnight_ms = (first_midnight - start_time).total_seconds() * 1000

    cached_pen_growth, cached_ante_growth = None, None

    while sim_elapsed < total_duration_ms:
        sim_elapsed += consts['PREDICTION_STEP_MS']
        if sim_elapsed > total_duration_ms: sim_elapsed = total_duration_ms
        sim_time = start_time + timedelta(milliseconds=sim_elapsed)

        if sim_elapsed >= end_phase_start_ms:
            phase = 'end'
        elif sim_elapsed >= penultimate_phase_start_ms:
            phase = 'penultimate'
        elif sim_elapsed < first_midnight_ms:
            phase = 'start'
        else:
            phase = 'middle'

        model, feats = models[phase], models['features'][phase]

        row = {}
        row['is_weekend'] = 1 if sim_time.weekday() >= 5 else 0
        today_midnight = sim_time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_progress = ((sim_time - today_midnight).total_seconds() * 1000) / consts['MS_PER_DAY']
        row['day_progress'] = day_progress
        row['hour_sin'] = np.sin(2 * np.pi * day_progress)
        row['hour_cos'] = np.cos(2 * np.pi * day_progress)

        time_until_end = total_duration_ms - sim_elapsed
        row['time_until_end_ms_norm'] = scalers['time_until_end_ms'].transform([[time_until_end]])[0][0]
        row['elapsed_ms_norm'] = scalers['elapsed_ms'].transform([[sim_elapsed]])[0][0]
        row['event_intensity_norm'] = current_intensity_norm

        if phase in ['end', 'penultimate']:
            raw_inv = 1.0 / (time_until_end / 3600000.0 + 0.25)
            row['inv_time_hr_norm'] = scalers['inv_time_hr'].transform([[raw_inv]])[0][0]

        pred_rate_raw = 0.0
        # X_in = pd.DataFrame([row])[feats] # 移除这一行，避免在lag生成前调用

        if phase == 'start':
            # 先生成 Lag 特征
            for i in range(1, get_config("WINDOW_SIZE") + 1):
                row[f'lag_{i}'] = buffer[-i]['pt_rate_norm'] if len(buffer) >= i else 0

            # 再生成 X_in
            X_in = pd.DataFrame([row])[feats]

            pred_rate_norm = model.predict(X_in)[0]
            pred_rate_raw = scalers['pt_rate'].inverse_transform([[pred_rate_norm]])[0][0]

        elif phase == 'end':
            # 生成 X_in
            X_in = pd.DataFrame([row])[feats]

            if cached_pen_growth is None:
                ref_end_ms = end_phase_start_ms
                ref_start_ms = ref_end_ms - (get_config("END_REFERENCE_HOURS") * 3600000)
                pt_ref_end = get_pt_at_elapsed(buffer, ref_end_ms)
                pt_ref_start = get_pt_at_elapsed(buffer, ref_start_ms)
                cached_pen_growth = max(pt_ref_end - pt_ref_start, 1000)

            pred_scaled_rate = model.predict(X_in)[0]
            avg_hourly_growth = cached_pen_growth / get_config("END_REFERENCE_HOURS")
            pred_rate_raw = pred_scaled_rate * avg_hourly_growth

        elif phase == 'penultimate':
            # 生成 X_in
            X_in = pd.DataFrame([row])[feats]

            if cached_ante_growth is None:
                ref_end_ms = penultimate_phase_start_ms
                ref_start_ms = ref_end_ms - (get_config("PENULTIMATE_REFERENCE_HOURS") * 3600000)
                pt_ref_end = get_pt_at_elapsed(buffer, ref_end_ms)
                pt_ref_start = get_pt_at_elapsed(buffer, ref_start_ms)
                cached_ante_growth = max(pt_ref_end - pt_ref_start, 1000)

            pred_scaled_rate = model.predict(X_in)[0]
            avg_hourly_growth = cached_ante_growth / get_config("PENULTIMATE_REFERENCE_HOURS")
            pred_rate_raw = pred_scaled_rate * avg_hourly_growth

        else:
            # Middle
            # 生成 X_in
            X_in = pd.DataFrame([row])[feats]

            rate_24h_raw = get_24h_ago_raw_value(buffer, sim_elapsed)
            if rate_24h_raw > 1e-9:
                pred_log_diff = model.predict(X_in)[0]
                pred_rate_raw = np.expm1(pred_log_diff + np.log1p(rate_24h_raw))
            else:
                pred_rate_raw = 0

        # 软速率限制
        max_rate_ms = get_config("MAX_HOURLY_SPEED") / 3600000.0
        if max_rate_ms > 0: pred_rate_raw = max_rate_ms * np.tanh(pred_rate_raw / max_rate_ms)
        if pred_rate_raw < 0: pred_rate_raw = 0

        curr_pt += pred_rate_raw * (sim_elapsed - buffer[-1]['elapsed_ms'])
        pred_pts.append(curr_pt)
        pred_times.append(sim_elapsed)

        pred_rate_norm = scalers['pt_rate'].transform([[pred_rate_raw]])[0][0]
        buffer.append({
            'elapsed_ms': sim_elapsed, 'pt(ep)': curr_pt,
            'pt_rate_norm': pred_rate_norm, 'pt_rate_raw': pred_rate_raw
        })

    return pred_pts, pred_times, total_duration_ms


# ==========================================
#           第五部分：兼容性适配器 (Server API)
# ==========================================

def preprocess_current_data(df, start_ts):
    consts = get_ms_constants()
    df = df.copy()
    df['elapsed_ms'] = df['time_point(ms)'] - start_ts
    df['pt_diff'] = df['pt(ep)'].diff().fillna(0)
    df['time_diff'] = df['elapsed_ms'].diff().fillna(consts['PREDICTION_STEP_MS'])
    df.loc[df['time_diff'] <= 0, 'time_diff'] = consts['PREDICTION_STEP_MS']
    df['pt_rate'] = df['pt_diff'] / df['time_diff']
    return df


def train_or_load_model(directory, target_event_type, target_event_id, progress_callback=None):
    models = build_segmented_models(directory, target_event_type, target_event_id, progress_callback)
    return models, None, models['scalers']


def predict_recursive(model, features, df_curr, scalers, end_time_str, target_event_id, progress_callback=None):
    meta = get_event_metadata(target_event_id)
    if not meta: raise ValueError("无法获取活动元数据")

    offset = df_curr['time_point(ms)'].iloc[0] - meta['start_ts']
    df_proc = preprocess_current_data(df_curr, meta['start_ts'])

    preds, times, dur = predict_relay(model, df_proc, meta, progress_callback)

    # 修复：如果 preds 为空（说明当前数据已包含结束时间，没有新的预测点）
    # 返回最后一个实际点作为“预测”结果，防止前端显示 0
    if not preds and not df_curr.empty:
        last_real_pt = df_curr['pt(ep)'].iloc[-1]
        preds = [last_real_pt]
        # time 使用实际结束时间或最后一个点的时间
        last_real_elapsed = df_curr['time_point(ms)'].iloc[-1] - meta['start_ts']
        times = [last_real_elapsed]

    adjusted_times = [t - offset for t in times]
    adjusted_total_dur = dur - offset
    return preds, adjusted_times, adjusted_total_dur