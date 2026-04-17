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

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 机器学习库
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

    # 核心模型参数 (LSTM)
    "WINDOW_SIZE": 16,  # LSTM 序列长度
    "EPOCHS": 50,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.005,
    "HIDDEN_SIZE": 32,
    "NUM_LAYERS": 1,

    # 约束条件
    "MAX_HOURLY_SPEED": 700000,
    
    # 损失函数参数
    "ASYM_UNDER_WEIGHT": 5.0
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
        "MS_PER_HOUR": 3600 * 1000,
        "MS_PER_DAY": 24 * 3600 * 1000
    }

# 路径常量
EVENT_DATA_DIR = '../event_data'  # 指向上级目录的数据，避免重复下载
MODEL_DIR = 'models_lstm'
TEMP_CURRENT_FILE = 'current_event_temp_lstm.csv'
CACHE_FILE = '../cache_data/event_types_cache.json'
LATEST_EVENT_CACHE_FILE = '../cache_data/latest_event_cache.json'
CURRENT_EVENT_INFO_FILE = 'current_event_info_lstm.json'

# 判断设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for LSTM: {device}")

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
            os.makedirs(os.path.dirname(LATEST_EVENT_CACHE_FILE), exist_ok=True)
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
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
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
    """
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
        if os.path.exists(TEMP_CURRENT_FILE):
            should_clean = True

    if should_clean:
        print(f"[Core] 检测到活动切换或缓存失效 (Target: {event_id})，清理旧数据文件...")
        try:
            if os.path.exists(TEMP_CURRENT_FILE):
                os.remove(TEMP_CURRENT_FILE)
        except Exception as e:
            print(f"[Core] 清理失败: {e}")

    try:
        with open(CURRENT_EVENT_INFO_FILE, 'w') as f:
            json.dump({'event_id': event_id, 'updated_at': time.time()}, f)
    except Exception as e:
        print(f"[Core] 保存活动信息失败: {e}")

    print(f"[Core] 正在尝试获取 Event {event_id} 数据...")
    max_retries = 5
    retry_delay = 2

    new_data_points = []
    success = False

    for attempt in range(max_retries):
        try:
            tracker = EventTracker(event=event_id, server=server)
            data = tracker.get_data(1000)

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

    existing_df = pd.DataFrame()
    if os.path.exists(TEMP_CURRENT_FILE):
        try:
            existing_df = pd.read_csv(TEMP_CURRENT_FILE)
        except Exception as e:
            print(f"[Core] Warning: 现有文件损坏或不可读, 将覆盖。 {e}")

    new_rows = []
    for point in new_data_points:
        dt = datetime.utcfromtimestamp(point['time'] / 1000) + timedelta(hours=8)
        new_rows.append({
            'time_point(ms)': point['time'],
            'time(UTC+8)': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'pt(ep)': point['ep']
        })
    new_df = pd.DataFrame(new_rows)

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['time_point(ms)'], keep='last', inplace=True)
        combined_df.sort_values(by='time_point(ms)', inplace=True)
    else:
        combined_df = new_df
        combined_df.sort_values(by='time_point(ms)', inplace=True)

    try:
        combined_df.to_csv(TEMP_CURRENT_FILE, index=False, encoding='utf-8')
        print(f"[Core] 数据已合并并保存，总点数: {len(combined_df)}")
        return True
    except Exception as e:
        print(f"[Core] 保存合并数据失败: {e}")
        return False


# ==========================================
#           第二部分：LSTM 架构与特征提取
# ==========================================

class AsymmetricLoss(nn.Module):
    """
    非对称损失函数 (Asymmetric Loss)。
    对于预测值低于真实值 (Under-prediction) 的情况施加更大的惩罚。
    这在预测活动分数线时至关重要，因为低估分数线会导致玩家意外掉档，而高估则相对安全。
    """
    def __init__(self, weight_under=5.0):
        super(AsymmetricLoss, self).__init__()
        self.weight_under = weight_under

    def forward(self, pred, target):
        error = target - pred
        # error > 0 表示预测值比真实值小，即低估 (Under-estimation)
        loss = torch.where(error > 0, self.weight_under * (error ** 2), error ** 2)
        return torch.mean(loss)

# ==========================================

class EventLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(EventLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # 只取序列最后一步的输出
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences(directory, target_event_type, target_event_id):
    consts = get_ms_constants()
    if target_event_type == 'all':
        files = glob.glob(os.path.join(directory, '*', 'event_*.csv'))
    else:
        files = glob.glob(os.path.join(directory, target_event_type, 'event_*.csv'))
    
    all_data = []

    print(f"正在进行LSTM特征提取... (Window={get_config('WINDOW_SIZE')})")

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

            start_time = df['time(UTC+8)'].iloc[0]
            total_ms = df['elapsed_ms'].iloc[-1]
            df['time_until_end_ms'] = total_ms - df['elapsed_ms']
            
            df['is_weekend'] = (df['time(UTC+8)'].dt.weekday >= 5).astype(int)
            normalized = df['time(UTC+8)'].dt.normalize()
            seconds = (df['time(UTC+8)'] - normalized).dt.total_seconds()
            df['day_progress'] = seconds * 1000 / consts['MS_PER_DAY']
            df['hour_sin'] = np.sin(2 * np.pi * df['day_progress'])
            df['hour_cos'] = np.cos(2 * np.pi * df['day_progress'])
            
            all_data.append(df)

        except Exception:
            continue
            
    return all_data

# ==========================================
#           第三部分：模型训练
# ==========================================

def build_segmented_models(directory, target_event_type, target_event_id, progress_callback=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    window_size = get_config("WINDOW_SIZE")
    hasher = hashlib.md5()
    config_state = f"{target_event_type}_lstm_v6_mono_{window_size}_{get_config('EPOCHS')}_{get_config('HIDDEN_SIZE')}_{get_config('ASYM_UNDER_WEIGHT')}"
    hasher.update(config_state.encode('utf-8'))
    model_hash = hasher.hexdigest()
    path_prefix = os.path.join(MODEL_DIR, f"{model_hash}")

    features = ['pt_rate_norm', 'is_weekend', 'hour_sin', 'hour_cos', 'elapsed_ms_norm', 'time_until_end_ms_norm', 'inv_time_hr_norm', 'event_intensity_norm']

    if os.path.exists(f"{path_prefix}_model.pth"):
        if progress_callback: progress_callback(70, "加载缓存的分段 LSTM 模型...")
        
        model = EventLSTM(input_size=len(features), hidden_size=get_config("HIDDEN_SIZE"), num_layers=get_config("NUM_LAYERS")).to(device)
        model.load_state_dict(torch.load(f"{path_prefix}_model.pth", map_location=device, weights_only=True))
        model.eval()

        return {
            'model': model,
            'scalers': joblib.load(f"{path_prefix}_scalers.pkl"),
            'features': joblib.load(f"{path_prefix}_features.pkl")
        }

    if progress_callback: progress_callback(20, "提取全大盘历史特征进行预训练...")
    dfs_all = prepare_sequences(directory, 'all', target_event_id)
    if not dfs_all: raise ValueError("没有足够大盘数据训练")

    big_df = pd.concat(dfs_all, ignore_index=True)

    # 训练 Scalers
    scaler_rate = MinMaxScaler(feature_range=(0, 1))
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
    
    def make_dataset(df_list):
        X_list, Y_list = [], []
        for df in df_list:
            df['pt_rate_norm'] = scalers['pt_rate'].transform(df['pt_rate'].values.reshape(-1, 1)).ravel()
            df['elapsed_ms_norm'] = scalers['elapsed_ms'].transform(df['elapsed_ms'].values.reshape(-1, 1)).ravel()
            df['time_until_end_ms_norm'] = scalers['time_until_end_ms'].transform(df['time_until_end_ms'].values.reshape(-1, 1)).ravel()
            
            df['inv_time_hr'] = 1.0 / (df['time_until_end_ms'] / 3600000.0 + 0.25)
            df['inv_time_hr_norm'] = scalers['inv_time_hr'].transform(df['inv_time_hr'].values.reshape(-1, 1)).ravel()
            
            df['event_intensity_norm'] = scalers['event_intensity'].transform(df['event_intensity'].values.reshape(-1, 1)).ravel()
            
            arr = df[features].values
            for i in range(len(arr) - window_size):
                X_list.append(arr[i:i+window_size])
                Y_list.append(arr[i+window_size, 0]) 
        if not X_list: return None
        X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
        y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32).view(-1, 1).to(device)
        return TensorDataset(X_tensor, y_tensor)

    dataset_pretrain = make_dataset(dfs_all)
    dataloader_pretrain = DataLoader(dataset_pretrain, batch_size=get_config("BATCH_SIZE"), shuffle=True)

    model = EventLSTM(input_size=len(features), hidden_size=get_config("HIDDEN_SIZE"), num_layers=get_config("NUM_LAYERS")).to(device)
    criterion = AsymmetricLoss(weight_under=get_config("ASYM_UNDER_WEIGHT"))
    optimizer = torch.optim.Adam(model.parameters(), lr=get_config("LEARNING_RATE"))

    epochs = get_config("EPOCHS")
    if progress_callback: progress_callback(40, f"开始混合预训练 (基于 {len(dfs_all)} 场活动)...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader_pretrain:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if progress_callback and epoch % 10 == 0:
            progress_callback(40 + int((epoch / epochs) * 20), f"Pre-train Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader_pretrain):.4f}")

    if target_event_type != 'all' and target_event_type != 'unknown':
        if progress_callback: progress_callback(60, f"提取 {target_event_type} 特定特征进行微调...")
        dfs_specific = prepare_sequences(directory, target_event_type, target_event_id)
        if dfs_specific:
            dataset_finetune = make_dataset(dfs_specific)
            if dataset_finetune:
                dataloader_finetune = DataLoader(dataset_finetune, batch_size=get_config("BATCH_SIZE"), shuffle=True)
                finetune_epochs = max(10, epochs // 2)
                optimizer_ft = torch.optim.Adam(model.parameters(), lr=get_config("LEARNING_RATE") / 2)
                
                if progress_callback: progress_callback(70, f"开始特定类型微调 (基于 {len(dfs_specific)} 场)...")
                for epoch in range(finetune_epochs):
                    epoch_loss = 0
                    for batch_X, batch_y in dataloader_finetune:
                        optimizer_ft.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer_ft.step()
                        epoch_loss += loss.item()
                    
                    if progress_callback and epoch % 10 == 0:
                        progress_callback(70 + int((epoch / finetune_epochs) * 20), f"Fine-tune Epoch {epoch}/{finetune_epochs}, Loss: {epoch_loss/len(dataloader_finetune):.4f}")

    model.eval()
    
    torch.save(model.state_dict(), f"{path_prefix}_model.pth")
    joblib.dump(scalers, f"{path_prefix}_scalers.pkl")
    joblib.dump(features, f"{path_prefix}_features.pkl")
    
    return {
        'model': model,
        'scalers': scalers,
        'features': features
    }

# ==========================================
#           第四部分：接力预测 (Buffer Relay)
# ==========================================

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
    if progress_callback: progress_callback(80, "开始 LSTM 接力预测...")

    start_time, end_time = meta['start_time'], meta['end_time']
    total_duration_ms = (end_time - start_time).total_seconds() * 1000
    scalers = models['scalers']
    model = models['model']
    features = models['features']
    
    window_size = get_config("WINDOW_SIZE")
    consts = get_ms_constants()

    df_curr['pt_rate_norm'] = scalers['pt_rate'].transform(df_curr['pt_rate'].values.reshape(-1, 1)).ravel()
    buffer = df_curr[['elapsed_ms', 'pt(ep)', 'pt_rate_norm', 'pt_rate']].rename(
        columns={'pt_rate': 'pt_rate_raw'}).to_dict('records')

    curr_elapsed = buffer[-1]['elapsed_ms']
    curr_pt = buffer[-1]['pt(ep)']
    
    curr_hours = curr_elapsed / consts['MS_PER_HOUR']
    current_intensity_raw = curr_pt / curr_hours if curr_hours > 0 else 0
    current_intensity_norm = scalers['event_intensity'].transform([[current_intensity_raw]])[0][0]

    pred_pts, pred_times = [], []
    sim_elapsed = curr_elapsed

    while sim_elapsed < total_duration_ms:
        sim_elapsed += consts['PREDICTION_STEP_MS']
        if sim_elapsed > total_duration_ms: sim_elapsed = total_duration_ms
        sim_time = start_time + timedelta(milliseconds=sim_elapsed)

        # 准备输入序列 (最后 window_size 个点)
        seq_data = []
        for i in range(window_size):
            idx = -window_size + i
            if len(buffer) + idx >= 0:
                item = buffer[len(buffer) + idx]
                item_time = start_time + timedelta(milliseconds=item['elapsed_ms'])
                day_progress = ((item_time - item_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000) / consts['MS_PER_DAY']
                time_until_end = total_duration_ms - item['elapsed_ms']
                inv_time_hr = 1.0 / (time_until_end / 3600000.0 + 0.25)
                
                row = [
                    item['pt_rate_norm'],
                    1 if item_time.weekday() >= 5 else 0,
                    np.sin(2 * np.pi * day_progress),
                    np.cos(2 * np.pi * day_progress),
                    scalers['elapsed_ms'].transform([[item['elapsed_ms']]])[0][0],
                    scalers['time_until_end_ms'].transform([[time_until_end]])[0][0],
                    scalers['inv_time_hr'].transform([[inv_time_hr]])[0][0],
                    current_intensity_norm
                ]
                seq_data.append(row)
            else:
                # 不足的数据用0填充
                seq_data.append([0]*len(features))
                
        # 预测
        X_in = torch.tensor([seq_data], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_rate_norm = model(X_in).item()
            
        pred_rate_raw = scalers['pt_rate'].inverse_transform([[pred_rate_norm]])[0][0]

        # 软速率限制
        max_rate_ms = get_config("MAX_HOURLY_SPEED") / 3600000.0
        
        # --- 动态指数重放 (Dynamic Exponentiation) 逻辑 ---
        # 即使使用了非对称 Loss 和 inv_time_hr，LSTM的本质仍然是回归均值（平滑曲线）
        # 在最后 24 小时，玩家的冲刺是处于非理性的指数级增长的
        # 我们在这里根据距离结束的时间，给推测出来的 pt_rate 加上一个动态放大倍数
        time_left_hours = (total_duration_ms - sim_elapsed) / consts['MS_PER_HOUR']
        if time_left_hours <= 24 and time_left_hours > 0:
            # 距离结束越近，放大倍数越高。最后 1 小时最多放大到 1.5 ~ 2.0 倍
            # 这是一个典型的倒数平滑曲线公式
            multiplier = 1.0 + (get_config("ASYM_UNDER_WEIGHT") / 10.0) * (1.0 - (time_left_hours / 24.0))**2
            pred_rate_raw = pred_rate_raw * multiplier

        if max_rate_ms > 0: pred_rate_raw = max_rate_ms * np.tanh(pred_rate_raw / max_rate_ms)
        if pred_rate_raw < 0: pred_rate_raw = 0

        curr_pt += pred_rate_raw * (sim_elapsed - buffer[-1]['elapsed_ms'])
        pred_pts.append(curr_pt)
        pred_times.append(sim_elapsed)

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


def predict_recursive(models, features_None, df_curr, scalers_None, end_time_str, target_event_id, progress_callback=None):
    meta = get_event_metadata(target_event_id)
    if not meta: raise ValueError("无法获取活动元数据")

    offset = df_curr['time_point(ms)'].iloc[0] - meta['start_ts']
    df_proc = preprocess_current_data(df_curr, meta['start_ts'])

    preds, times, dur = predict_relay(models, df_proc, meta, progress_callback)

    if not preds and not df_curr.empty:
        last_real_pt = df_curr['pt(ep)'].iloc[-1]
        preds = [last_real_pt]
        last_real_elapsed = df_curr['time_point(ms)'].iloc[-1] - meta['start_ts']
        times = [last_real_elapsed]

    adjusted_times = [t - offset for t in times]
    adjusted_total_dur = dur - offset
    return preds, adjusted_times, adjusted_total_dur