import os
import threading
import uuid
import time
import json
import csv
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

import event_predictor_lstm as core

app = Flask(__name__)

# --- 配置区域 ---
app.config['EVENT_DATA_DIR'] = 'event_data'
app.config['TEMP_CURRENT_FILE'] = 'current_event_temp.csv'
app.config['CACHE_DIR'] = 'cache_data'
app.config['TASKS_DIR'] = 'tasks'

os.makedirs(app.config['EVENT_DATA_DIR'], exist_ok=True)
os.makedirs(app.config['CACHE_DIR'], exist_ok=True)
os.makedirs(app.config['TASKS_DIR'], exist_ok=True)

MONITOR_STARTED = False


# ==========================================
#           文件持久化辅助函数
# ==========================================

def save_json(filepath, data):
    try:
        temp_path = filepath + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, filepath)
    except Exception as e:
        print(f"[File Error] Save failed {filepath}: {e}")


def load_json(filepath, default=None):
    if not os.path.exists(filepath): return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[File Error] Load failed {filepath}: {e}")
        return default


# --- 任务 & 状态管理 ---

def get_task_path(task_id): return os.path.join(app.config['TASKS_DIR'], f"{task_id}.json")


def save_task_state(task_id, state_data): save_json(get_task_path(task_id), state_data)


def get_task_state(task_id): return load_json(get_task_path(task_id))


def get_monitor_state_path(): return os.path.join(app.config['CACHE_DIR'], 'monitor_state.json')


def save_monitor_timestamp(timestamp): save_json(get_monitor_state_path(),
                                                 {'last_processed_timestamp': timestamp, 'updated_at': time.time()})


def get_monitor_timestamp(): return load_json(get_monitor_state_path(), default={}).get('last_processed_timestamp',
                                                                                        None)


def get_history_file_path(event_id): return os.path.join(app.config['CACHE_DIR'], f'history_preds_{event_id}.csv')


def get_latest_result_file_path(event_id): return os.path.join(app.config['CACHE_DIR'],
                                                               f'latest_result_{event_id}.json')


def append_prediction_history(event_id, prediction_time_ts, predicted_final_pt):
    file_path = get_history_file_path(event_id)
    file_exists = os.path.exists(file_path)
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['prediction_timestamp', 'predicted_final_pt'])
            writer.writerow([prediction_time_ts, int(predicted_final_pt)])
    except Exception as e:
        print(f"Error appending history: {e}")


def load_prediction_history(event_id):
    file_path = get_history_file_path(event_id)
    history_data = []
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                history_data.append({'x': int(row['prediction_timestamp']), 'y': int(row['predicted_final_pt'])})
        except Exception as e:
            print(f"Error loading history: {e}")
    return history_data


def get_current_data_timestamp():
    if not os.path.exists(app.config['TEMP_CURRENT_FILE']): return 0
    try:
        with open(app.config['TEMP_CURRENT_FILE'], 'rb') as f:
            try:
                f.seek(-1024, os.SEEK_END)
            except OSError:
                pass
            last_lines = f.readlines()
            if not last_lines: return 0
            parts = last_lines[-1].decode('utf-8').strip().split(',')
            if len(parts) >= 1 and parts[0].isdigit(): return int(parts[0])
    except:
        pass
    return 0


# ==========================================
#           业务逻辑
# ==========================================

def perform_prediction_logic(event_id, cutoff_percentage=100, progress_callback=None):
    """
    运行完整的机器学习预测逻辑
    """
    if progress_callback is None: progress_callback = lambda p, m: print(f"[Core] {p}% - {m}")

    core.load_config()
    meta = core.get_event_metadata(event_id)
    if not meta: raise ValueError('Meta Error')

    core.ensure_historical_data(meta['type'], event_id, progress_callback=progress_callback)

    if not os.path.exists(core.TEMP_CURRENT_FILE):
        raise ValueError('本地数据文件缺失')

    current_data_ts = get_current_data_timestamp()

    # 训练/加载模型
    model, features, scalers = core.train_or_load_model(
        core.EVENT_DATA_DIR, meta['type'], event_id, progress_callback=progress_callback
    )

    try:
        df_curr_full = pd.read_csv(core.TEMP_CURRENT_FILE)
    except Exception:
        raise ValueError('本地数据文件读取失败')

    df_curr_full['time(UTC+8)'] = pd.to_datetime(df_curr_full['time(UTC+8)'])

    df_input = df_curr_full
    if cutoff_percentage < 100:
        total_rows = len(df_curr_full)
        cut_idx = max(int(total_rows * (cutoff_percentage / 100.0)), core.get_config("WINDOW_SIZE") + 5)
        if cut_idx < total_rows:
            df_input = df_curr_full.iloc[:cut_idx].copy()
            progress_callback(60, f"Debug: Using {cutoff_percentage}% data...")
    else:
        df_input = df_curr_full.copy()

    # 递归预测
    preds, elapsed_ms, total_dur = core.predict_recursive(
        model, features, df_input, scalers, meta['end_time_str'], event_id, progress_callback=progress_callback
    )

    final_pt = preds[-1] if preds else 0
    progress_callback(99, "整理结果...")
    base_ts = df_curr_full['time_point(ms)'].iloc[0]

    actual_data = [{'x': int(r['time_point(ms)']), 'y': int(r['pt(ep)'])} for _, r in df_curr_full.iterrows()]
    start_ts = int(meta['start_ts'])
    if not actual_data or actual_data[0]['x'] > start_ts:
        actual_data.insert(0, {'x': start_ts, 'y': 0})
    elif actual_data:
        actual_data[0]['y'] = 0

    predict_data = []
    if not df_input.empty:
        last_pt = df_input.iloc[-1]
        predict_data.append({'x': int(last_pt['time_point(ms)']), 'y': int(last_pt['pt(ep)'])})

    for pt, elapsed in zip(preds, elapsed_ms):
        predict_data.append({'x': int(base_ts + elapsed), 'y': int(pt)})

    return {
        'event_id': event_id, 'event_name': meta['name'], 'event_type': meta['type'],
        'final_pt': int(final_pt), 'updated_at': int(time.time() * 1000),
        'data_timestamp': current_data_ts,
        'is_ended': False,
        'chart_data': {'actual': actual_data, 'predict': predict_data}
    }


def generate_static_result(event_id):
    """
    当活动已结束时调用：只读取实际数据，不运行预测，直接返回最终结果
    """
    meta = core.get_event_metadata(event_id)
    if not meta: raise ValueError('Meta Error')

    if not os.path.exists(core.TEMP_CURRENT_FILE):
        raise ValueError('本地数据文件缺失')

    df = pd.read_csv(core.TEMP_CURRENT_FILE)

    # 获取实际最终值
    final_pt = df['pt(ep)'].iloc[-1] if not df.empty else 0
    current_data_ts = get_current_data_timestamp()

    actual_data = [{'x': int(r['time_point(ms)']), 'y': int(r['pt(ep)'])} for _, r in df.iterrows()]
    start_ts = int(meta['start_ts'])
    if not actual_data or actual_data[0]['x'] > start_ts:
        actual_data.insert(0, {'x': start_ts, 'y': 0})
    elif actual_data:
        actual_data[0]['y'] = 0

    # 预测数据为空
    predict_data = []

    return {
        'event_id': event_id, 'event_name': meta['name'], 'event_type': meta['type'],
        'final_pt': int(final_pt), 'updated_at': int(time.time() * 1000),
        'data_timestamp': current_data_ts,
        'is_ended': True,  # 标记活动已结束
        'chart_data': {'actual': actual_data, 'predict': predict_data}
    }


# ==========================================
#           后台任务与线程
# ==========================================

def background_monitor_task():
    print(">>> Monitor Thread Started...")
    while True:
        try:
            event_id = core.get_latest_event_id()
            if not event_id:
                time.sleep(600)
                continue

            # === 检测活动是否结束 ===
            meta = core.get_event_metadata(event_id)
            if not meta:
                # 获取元数据失败，等待后重试，不要继续执行
                print("[Monitor] 获取元数据失败，跳过...")
                time.sleep(60)
                continue

            now_ts = int(time.time() * 1000)
            # 如果当前时间已超过结束时间，完全跳过自动任务
            if now_ts > meta['end_ts']:
                print(f"[Monitor] 活动 {event_id} 已结束，停止自动预测和记录。")
                time.sleep(600)
                continue
            # =====================

            last_processed_ts = get_monitor_timestamp()

            # 1. 尝试获取数据（内含 Retry & Merge）
            core.fetch_current_event_data(event_id)

            current_ts = get_current_data_timestamp()
            should_run = False

            if current_ts > 0:
                if last_processed_ts is None or current_ts > last_processed_ts:
                    print(f"[Monitor] New data (TS: {current_ts}), predicting...")
                    should_run = True

            cache_file = get_latest_result_file_path(event_id)
            if not os.path.exists(cache_file): should_run = True

            if should_run:
                try:
                    # 2. 运行预测
                    result = perform_prediction_logic(event_id, 100, lambda p, m: print(f"[Auto] {p}% {m}"))

                    save_monitor_timestamp(result['data_timestamp'])
                    save_json(cache_file, result)

                    # 3. 写入历史（双重校验）
                    # 校验 A: 时间未结束
                    # 校验 B: 预测值 > 0 (防止写入 0 或异常值)
                    current_final_pt = int(result['final_pt'])

                    if current_final_pt > 0 and int(time.time() * 1000) <= meta['end_ts']:
                        append_prediction_history(event_id, int(time.time() * 1000), current_final_pt)
                        print(f"[Monitor] Done. Final: {current_final_pt}")
                    else:
                        print(f"[Monitor] 预测结果 {current_final_pt} 无效或活动已结束，跳过历史记录。")

                except Exception as e:
                    print(f"[Monitor] Predict failed: {e}")

        except Exception as e:
            print(f"[Monitor] Loop error: {e}")
        time.sleep(600)


def run_prediction_task_wrapper(task_id, event_id, cutoff_percentage=100):
    """
    前端手动触发的任务
    """
    task_state = {'status': 'pending', 'progress': 0, 'message': 'Queued...', 'result': None, 'error': None}
    save_task_state(task_id, task_state)
    try:
        task_state.update({'status': 'running', 'progress': 1})
        save_task_state(task_id, task_state)

        # 检查活动状态
        meta = core.get_event_metadata(event_id)
        is_ended = False
        if meta and int(time.time() * 1000) > meta['end_ts']:
            is_ended = True

        if cutoff_percentage == 100:
            # 1. 下载/同步最新数据
            core.fetch_current_event_data(event_id)

            current_data_ts = get_current_data_timestamp()
            cache_file = get_latest_result_file_path(event_id)
            cached_result = load_json(cache_file)

            # 如果缓存命中
            if cached_result and cached_result.get('data_timestamp', 0) >= current_data_ts and current_data_ts > 0:
                history_data = load_prediction_history(event_id)
                cached_result['chart_data']['history_preds'] = history_data
                task_state.update(
                    {'result': cached_result, 'status': 'completed', 'progress': 100, 'message': 'From Cache'})
                save_task_state(task_id, task_state)
                return

        def update_progress(pct, msg):
            task_state.update({'progress': pct, 'message': msg})
            save_task_state(task_id, task_state)

        update_progress(10, "Initializing...")

        # 2. 根据活动状态选择逻辑
        if is_ended and cutoff_percentage == 100:
            # === 活动已结束，并且没有使用截断滑块（非回测） ===
            # 不运行预测模型，直接生成静态结果
            update_progress(50, "活动已结束，生成最终报告...")
            result_data = generate_static_result(event_id)
            update_progress(100, "Done")

            # 缓存结果以便显示，但【不】写入 append_prediction_history
            cache_file = get_latest_result_file_path(event_id)
            save_json(cache_file, result_data)
        else:
            # === 活动进行中 或 手动截断回测已结束的活动 ===
            # 运行完整预测
            result_data = perform_prediction_logic(event_id, cutoff_percentage, update_progress)

            if cutoff_percentage == 100:
                update_progress(95, "Caching...")
                cache_file = get_latest_result_file_path(event_id)
                save_json(cache_file, result_data)
                save_monitor_timestamp(result_data['data_timestamp'])

                # 写入历史记录 (增加 > 0 校验)
                if result_data['final_pt'] > 0:
                    append_prediction_history(event_id, int(time.time() * 1000), result_data['final_pt'])

        result_data['chart_data']['history_preds'] = load_prediction_history(event_id)
        task_state.update({'result': result_data, 'status': 'completed', 'progress': 100, 'message': 'Done!'})
        save_task_state(task_id, task_state)

    except Exception as e:
        import traceback;
        traceback.print_exc()
        task_state.update({'status': 'failed', 'error': str(e)})
        save_task_state(task_id, task_state)


# ==========================================
#           Flask 路由
# ==========================================

@app.route('/')
def index(): return render_template('index_lstm.html')


@app.route('/current_event', methods=['GET'])
def get_current_event():
    eid = core.get_latest_event_id()
    return jsonify({'success': True, 'event_id': eid}) if eid else jsonify({'success': False, 'error': 'Not found'})


@app.route('/latest_result', methods=['GET'])
def get_latest_result():
    event_id = request.args.get('event_id') or core.get_latest_event_id()
    if not event_id: return jsonify({'success': False, 'error': 'No Event ID'})
    data = load_json(get_latest_result_file_path(event_id))
    if data:
        data['chart_data']['history_preds'] = load_prediction_history(event_id)
        return jsonify({'success': True, 'result': data})
    return jsonify({'success': False, 'error': 'No cache'})


@app.route('/predict', methods=['POST'])
def start_predict():
    data = request.json
    task_id = str(uuid.uuid4())
    t = threading.Thread(target=run_prediction_task_wrapper,
                         args=(task_id, int(data.get('event_id')), int(data.get('cutoff_percentage', 100))))
    t.daemon = True
    t.start()
    return jsonify({'success': True, 'task_id': task_id})


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = get_task_state(task_id)
    return jsonify({'success': True, 'status': task['status'], 'progress': task['progress'], 'message': task['message'],
                    'result': task.get('result'), 'error': task.get('error')}) if task else jsonify(
        {'success': False, 'error': 'Not found'})


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(core.load_config())


@app.route('/api/settings', methods=['POST'])
def update_settings():
    new_config = request.json
    if core.save_config(new_config):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Failed to save'}), 500


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        if not MONITOR_STARTED:
            threading.Thread(target=background_monitor_task, daemon=True).start()
            MONITOR_STARTED = True
    app.run(host='0.0.0.0', port=5001, debug=False)