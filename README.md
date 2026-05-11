# bangDream_myycx1000

BanG Dream! Girls Band Party 活动分数线预测项目。项目会从 Bestdori 获取活动与榜线数据，基于历史同类型活动训练模型，对当前活动最终 PT 进行预测，并提供 Web 页面和 HTTP API 查询能力。

## 功能特性

- 自动识别当前最新活动，并拉取 Bestdori 活动榜线数据。
- 支持两套预测服务：
  - `lightGM_version`：基于 LightGBM 的分阶段预测模型。
  - `lstm_version`：基于 PyTorch LSTM 的序列预测模型。
- Web 页面展示实际曲线、预测曲线和历史预测记录。
- 后台定时监控，每 10 分钟尝试同步新数据并更新预测缓存。
- 支持 Docker Compose 一键启动 LightGBM 和 LSTM 服务。

## 项目结构

```text
.
├── docker-compose.yml          # 根目录编排：LightGBM / LSTM
├── requirements.txt            # 根目录汇总依赖
├── event_data/                 # 历史活动数据
├── cache_data/                 # 缓存数据
├── lightGM_version/
│   ├── server2.py              # LightGBM Flask 服务入口，端口 5000
│   ├── event_predictor_auto3.py
│   ├── templates/index.html
│   ├── models_overlap/         # LightGBM 模型缓存
│   └── Dockerfile
├── lstm_version/
│   ├── server_lstm.py          # LSTM Flask 服务入口，端口 5001
│   ├── event_predictor_lstm.py
│   ├── templates/index_lstm.html
│   ├── models_lstm/            # LSTM 模型缓存
│   └── Dockerfile
```

## 环境要求

- Python 3.10 推荐。
- Docker / Docker Compose 可选，用于容器化部署。
- 运行预测时需要能访问 Bestdori API。
- LSTM 默认使用 CPU 版 PyTorch；如果本机有 CUDA，可按需自行安装对应版本。

## 快速启动

### 使用 Docker Compose

在项目根目录执行：

```bash
docker compose up -d --build
```

启动后默认服务：

- LightGBM Web：`http://localhost:5000`
- LSTM Web：`http://localhost:5001`

查看日志：

```bash
docker compose logs -f lightgm
docker compose logs -f lstm
```

停止服务：

```bash
docker compose down
```

### 本地运行 LightGBM 版本

```bash
cd lightGM_version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server2.py
```

访问 `http://localhost:5000`。

### 本地运行 LSTM 版本

```bash
cd lstm_version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server_lstm.py
```

访问 `http://localhost:5001`。

## Web 与 API

两个预测服务的接口基本一致。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | Web 页面 |
| `GET` | `/current_event` | 获取当前最新活动 ID |
| `GET` | `/latest_result?event_id=295` | 获取缓存中的最新预测结果 |
| `POST` | `/predict` | 创建预测任务 |
| `GET` | `/status/<task_id>` | 查询预测任务进度与结果 |
| `GET` | `/api/settings` | 获取模型配置 |
| `POST` | `/api/settings` | 更新模型配置 |

创建预测任务示例：

```bash
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"event_id\":295,\"cutoff_percentage\":100}"
```

返回 `task_id` 后查询状态：

```bash
curl http://localhost:5000/status/<task_id>
```

`cutoff_percentage` 可用于回测，例如传入 `50` 表示只使用当前活动前 50% 的数据进行预测。

## 配置说明

模型配置通过各版本目录下运行时生成的 `config.json` 管理，也可以通过 `/api/settings` 接口读取和更新。

LightGBM 主要配置：

- `TARGET_EVENT_ID`：目标活动 ID。
- `HISTORY_CHECK_RANGE`：向前扫描历史活动的范围。
- `MIN_HISTORY_EVENT_ID`：最小历史活动 ID。
- `PREDICTION_STEP_MINUTES`：预测步长，默认 30 分钟。
- `WINDOW_SIZE`：滞后特征窗口。
- `OVERLAP_HOURS`：切片重叠时长。
- `MAX_HOURLY_SPEED`：小时速度软上限。
- `ASYM_*`：非对称损失权重，用于控制低估和高估惩罚。

LSTM 额外配置：

- `EPOCHS`：训练轮数。
- `BATCH_SIZE`：批大小。
- `LEARNING_RATE`：学习率。
- `HIDDEN_SIZE`：隐藏层维度。
- `NUM_LAYERS`：LSTM 层数。

修改配置后，模型缓存哈希会变化，下一次预测可能触发重新训练。

## 数据与缓存

运行过程中会生成或更新以下文件：

- `event_data/<event_type>/event_<id>.csv`：历史活动榜线数据。
- `current_event_temp.csv` / `current_event_temp_lstm.csv`：当前活动临时数据。
- `cache_data/latest_result_<event_id>.json`：最新预测结果缓存。
- `cache_data/history_preds_<event_id>.csv`：历史预测记录。
- `tasks/<task_id>.json`：异步任务状态。
- `models_overlap/`、`models_lstm/`：训练好的模型与 scaler 缓存。

这些文件会影响后续预测速度和缓存命中情况。删除模型缓存后，下一次预测会重新训练。

## 常见问题

### 首次预测很慢

首次运行需要扫描历史活动、下载数据并训练模型。后续会复用 `event_data` 和模型缓存，速度会明显提升。

### `/latest_result` 返回 `No cache`

说明当前活动还没有完成过预测。先在 Web 页面点击预测，或调用 `/predict` 创建任务并等待完成。

### 无法获取最新活动

预测服务依赖 Bestdori API。请检查网络是否可访问 `bestdori.com`，以及当前服务器时间是否正确。

## 说明

预测结果仅供参考。活动最终分数线会受玩家行为、活动类型、卡池热度、节假日和临近结束冲刺等因素影响，模型输出不应作为唯一决策依据。

## 已知缺陷

- 目前活动前期预测普遍偏高，主要原因是调整了惩罚函数后模型更倾向于避免低估。
- 当前模型预测最准确的时间段大约在活动结束前 22 小时到 46 小时。
- 进入最后一天后，预测结果通常会出现约 5% 到 10% 的偏低。
