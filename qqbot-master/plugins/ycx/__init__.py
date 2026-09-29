import httpx
from nonebot import get_plugin_config, on_command
from nonebot.plugin import PluginMetadata
from nonebot.adapters.qq import Bot
from nonebot.adapters import Event
from nonebot.exception import FinishedException

from .config import Config

from nonebot.adapters.qq import MessageSegment
from .draw import create_prediction_poster

__plugin_meta__ = PluginMetadata(
    name="YCX Predictor",
    description="自动获取活动事件并提交预测任务",
    usage="/ycx",
    config=Config,
)

plugin_config = get_plugin_config(Config)

# 注册命令 /ycx
ycx_matcher = on_command(
    "ycx",
    priority=10,
    block=True
)


@ycx_matcher.handle()
async def handle_ycx(bot: Bot, event: Event):
    base_url = plugin_config.ycx_api_url

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{base_url}/qq_predict")

            if resp.status_code != 200:
                await ycx_matcher.finish(f"请求预测接口失败，HTTP状态码: {resp.status_code}")
                return

            data = resp.json()

            if 'final_pt' in data:
                try:
                    # 生成图片流
                    image_buf = await create_prediction_poster(data)
                    image_bytes = image_buf.getvalue()
                    
                    # 发送图片
                    await ycx_matcher.finish(MessageSegment.file_image(image_bytes))
                except FinishedException:
                    raise
                except Exception as draw_e:
                    import traceback
                    traceback.print_exc()
                    await ycx_matcher.finish(f"图片生成失败: {draw_e}\n预测结果: {data['final_pt']} PT")
            else:
                await ycx_matcher.finish(f"预测完成，但在返回数据中未找到结果。\n原始返回: {data}")

        except FinishedException:
            raise
        except httpx.RequestError as e:
            await ycx_matcher.finish(f"网络请求错误: {e}")
        except Exception as e:
            await ycx_matcher.finish(f"发生未知错误: {e}")