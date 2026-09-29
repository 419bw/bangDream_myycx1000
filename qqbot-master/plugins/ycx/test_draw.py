import asyncio
import time
from draw import create_prediction_poster
import random

async def main():
    # 构造假数据
    now_ms = int(time.time() * 1000)
    end_ms = now_ms + 86400 * 3 * 1000 # 3天后结束
    
    # 构建假的折线数据
    actual = []
    predict = []
    
    start_ms = now_ms - 86400 * 3 * 1000
    pt = 1000000
    for i in range(20):
        t = start_ms + i * 3600 * 4 * 1000
        pt += random.randint(10000, 50000)
        actual.append({'x': t, 'y': pt})
        
    p_pt = pt
    t = now_ms
    for i in range(20):
        predict.append({'x': t, 'y': p_pt})
        t += 3600 * 4 * 1000
        p_pt += random.randint(10000, 50000)
        
    data_dict = {
        'event_id': '302', # 使用一个存在的国服活动ID
        'final_pt': p_pt,
        'current_pt': pt,
        'speed': 25612,
        'updated_at': now_ms,
        'end_time': end_ms,
        'chart_data': {
            'actual': actual,
            'predict': predict
        }
    }
    
    print("生成中...")
    start_time = time.time()
    try:
        buf = await create_prediction_poster(data_dict)
        with open('test_output.png', 'wb') as f:
            f.write(buf.getvalue())
        print(f"生成成功! 测试图像已保存为 test_output.png. 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"生成失败: {e}")

if __name__ == '__main__':
    asyncio.run(main())
