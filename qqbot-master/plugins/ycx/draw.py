import httpx
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import colorsys
import random
import numpy as np

from bestdori.events import Event

# --- 辅助字体加载 ---
def get_font(size: int, bold=False):
    font_paths = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", # 微软雅黑
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", # Linux 文泉驿微米黑
        "C:/Windows/Fonts/simhei.ttf" # 黑体备用
    ]
    for p in font_paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()

# --- 核心突破：AI 色彩量化与自适应主题引擎 ---
def get_theme_colors(img: Image.Image):
    """提取图片主色并生成主题调色盘"""
    # 极速缩小至 50x15 像素
    small_img = img.copy().resize((50, 15), Image.Resampling.LANCZOS).convert("RGB")
    pixels = list(small_img.getdata())
    
    color_counts = {}
    for r, g, b in pixels:
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        # 剔除过暗或黑白灰
        if l < 0.2 or s < 0.2 or l > 0.95:
            continue
        
        # 简单量化（降低精度以合并相近颜色）
        bucket = (r // 20, g // 20, b // 20)
        color_counts[bucket] = color_counts.get(bucket, {"count": 0, "rgb": (0,0,0)})
        color_counts[bucket]["count"] += 1
        
        # 累计RGB值以计算平均值
        curr_count = color_counts[bucket]["count"]
        curr_r, curr_g, curr_b = color_counts[bucket]["rgb"]
        color_counts[bucket]["rgb"] = (
            (curr_r * (curr_count - 1) + r) // curr_count,
            (curr_g * (curr_count - 1) + g) // curr_count,
            (curr_b * (curr_count - 1) + b) // curr_count
        )

    if color_counts:
        # 找到频率最高的颜色桶
        best_bucket = max(color_counts.values(), key=lambda x: x["count"])
        r, g, b = best_bucket["rgb"]
    else:
        # 默认元气粉
        r, g, b = (255, 64, 129)
        
    # 生成调色盘
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    
    theme_main = (r, g, b, 255)
    
    # 浅柔背景色：稍微降低一点亮度，使得白色波点更加明显
    light_l = 0.92
    light_s = min(s, 0.4)
    lr, lg, lb = colorsys.hls_to_rgb(h, light_l, light_s)
    theme_light = (int(lr*255), int(lg*255), int(lb*255), 255)
    
    # 硬投影色：在此基础上加深
    shadow_l = 0.86
    shadow_s = s
    sr, sg, sb = colorsys.hls_to_rgb(h, shadow_l, shadow_s)
    theme_shadow = (int(sr*255), int(sg*255), int(sb*255), 255)
    
    def rgb2hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    return theme_main, theme_light, theme_shadow, rgb2hex(theme_main)


async def fetch_banner_bytes(event_id: str) -> bytes:
    try:
        event = Event(int(event_id))
        img_bytes = event.get_banner('cn')
        if not img_bytes:
             raise ValueError("Failed to fetch banner image using bestdori API")
        return img_bytes
    except Exception as e:
        raise ValueError(f"Bestdori API Error: {str(e)}")


def create_pop_background(theme_light, target_size=(1080, 1920)):
    """第一层：元气环境底板 (The Pop Base Layer)"""
    bg = Image.new('RGBA', target_size, theme_light)
    overlay = Image.new('RGBA', target_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 波点/气泡点缀
    random.seed(42) # 可固定种子或使用当前时间
    for _ in range(25):
        # 倾向于在四周生成
        x = random.choice([random.randint(-50, 300), random.randint(700, 1130)])
        y = random.choice([random.randint(-50, 400), random.randint(1500, 1970)])
        r = random.randint(20, 100)
        
        # 纯白，透明度约 40% (255 * 0.4 ≈ 102)
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 255, 255, 102))
        
    return Image.alpha_composite(bg, overlay)


def draw_header_card(bg, original_banner, theme_shadow, theme_main, event_id):
    """第二层：头部活动名片 (The Header Card)"""
    draw = ImageDraw.Draw(bg)
    
    # 软投影 Banner 容器
    # 1. 硬投影层 (x: 50 + 15, y: 80 + 15)
    draw.rounded_rectangle([(65, 95), (1045, 435)], radius=25, fill=theme_shadow)
    
    # 2. 真实毛玻璃外框层
    # 2.1 拷贝底图并进行高斯模糊
    blur_bg = bg.copy().filter(ImageFilter.GaussianBlur(radius=25))
    mask = Image.new('L', bg.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(50, 80), (1030, 420)], radius=25, fill=255)
    bg.paste(blur_bg, (0,0), mask=mask)
    
    # 2.2 覆盖半透明白色与描边 (Alpha=180)
    overlay = Image.new('RGBA', bg.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle([(50, 80), (1030, 420)], radius=25, fill=(255, 255, 255, 180), outline=(255, 255, 255, 200), width=2)
    bg = Image.alpha_composite(bg, overlay)
    
    # 更新 draw 对象
    draw = ImageDraw.Draw(bg)
    
    # 3. Banner 植入
    ow, oh = original_banner.size
    # 目标宽度为 940，等比缩放
    target_bw = 940
    scale = target_bw / ow
    target_bh = int(oh * scale)
    
    resized_banner = original_banner.resize((target_bw, target_bh), Image.Resampling.LANCZOS)
    
    # 为了完美居中，我们需要切出大概 300 的高度，如果不够就用全高
    final_bh = min(target_bh, 300)
    crop_top = (target_bh - final_bh) // 2
    banner_cropped = resized_banner.crop((0, crop_top, target_bw, crop_top + final_bh))
    
    # 处理圆角 (radius 25)
    mask = Image.new('L', banner_cropped.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), banner_cropped.size], radius=25, fill=255)
    
    banner_cropped.putalpha(mask)
    
    # 贴上去
    bg.paste(banner_cropped, (70, 100), mask=banner_cropped)
    
    # 绘制活动 ID 胶囊 (位置与数据卡片左侧靠齐 x=80，跨越白框边缘 y=65)
    font_capsule_large = get_font(30, bold=True)
    draw_capsule(draw, 90, 30, f"ID : {event_id}", font_capsule_large, bg_color=theme_main, height=56)
    
    return bg


def draw_capsule(draw, x, y, text, font, bg_color, text_color=(255,255,255,255), width=None, height=44):
    """绘制两端半圆的胶囊标签"""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    h = height # 胶囊高度
    w = width if width else (text_w + 50) # 胶囊宽度
    r = h // 2
    
    draw.rounded_rectangle([(x, y), (x+w, y+h)], radius=r, fill=bg_color)
    draw.text((x + (w - text_w)//2, y + (h - text_h)//2 - 4), text, font=font, fill=text_color)


def draw_data_board(bg, data_dict, theme_main, theme_shadow):
    """第三层：大圆角数据视窗 (双卡片拆分)"""
    draw = ImageDraw.Draw(bg)
    
    # 四象限独立卡片尺寸与起始系
    w_card = 480
    lx = 50
    rx = 550
    
    top_capsule_y = 450
    top_card_y1 = top_capsule_y + 22 # 高度 44 的一半
    top_card_y2 = top_card_y1 + 95 # 卡片高 95 像素
    
    bot_capsule_y = 580
    bot_card_y1 = bot_capsule_y + 22 # 高度 44 的一半
    bot_card_y2 = bot_card_y1 + 95 # 卡片高 95 像素
    
    # 局部真实毛玻璃画板函数
    def draw_bg_card(x1, y1, x2, y2):
        nonlocal bg, draw
        # 画阴影
        draw.rounded_rectangle([(x1+8, y1+8), (x2+8, y2+8)], radius=25, fill=theme_shadow)
        
        # 将被遮盖部分的底层背景模糊化
        blur_bg = bg.copy().filter(ImageFilter.GaussianBlur(radius=25))
        mask = Image.new('L', bg.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=25, fill=255)
        bg.paste(blur_bg, (0,0), mask=mask)
        
        # 增加透明度和白色薄描边实现毛玻璃质感 (Overlay 叠加)
        overlay = Image.new('RGBA', bg.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=25, fill=(255, 255, 255, 180), outline=(255, 255, 255, 200), width=2)
        
        bg = Image.alpha_composite(bg, overlay)
        draw = ImageDraw.Draw(bg)
        
    # 画底板 (带阴影)
    draw_bg_card(lx, top_card_y1, lx+w_card, top_card_y2) # 左上
    draw_bg_card(lx, bot_card_y1, lx+w_card, bot_card_y2) # 左下
    draw_bg_card(rx, top_card_y1, rx+w_card, top_card_y2) # 右上
    draw_bg_card(rx, bot_card_y1, rx+w_card, bot_card_y2) # 右下
    
    # 加载字体
    font_capsule_large = get_font(30, bold=True)
    font_capsule_med = get_font(22, bold=True)
    # 取消数据字体的加粗，使用更纤长圆润的常规体
    font_value_large = get_font(60, bold=False)
    font_value_med = get_font(44, bold=False)
    
    predict_pt = data_dict.get('final_pt', 0)
    current_pt = data_dict.get('current_pt', 0)
    speed_pt = data_dict.get('speed', 0)
    # 计算剩余时间
    event_end = data_dict.get('end_time', 0)
    now_ts = data_dict.get('updated_at', 0)
    
    remain_str = "活动已结束"
    if event_end > now_ts:
        seconds = (event_end - now_ts) / 1000
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        mins = int((seconds % 3600) // 60)
        
        if days > 0:
            remain_str = f"{days}天 {hours}小时 {mins}分"
        else:
            remain_str = f"{hours}小时 {mins}分"

    # -- 左侧阵列 --
    # 预测线 (统一使用中号胶囊与字体)
    draw_capsule(draw, lx+30, top_capsule_y, "预测线", font_capsule_med, bg_color=(96, 125, 139, 255))
    draw.text((lx+30, top_card_y1 + 25), f"{int(predict_pt):,}", font=font_value_med, fill=(51, 51, 51, 255))
    
    # 最新分数线
    draw_capsule(draw, lx+30, bot_capsule_y, "最新分数线", font_capsule_med, bg_color=(96, 125, 139, 255))
    draw.text((lx+30, bot_card_y1 + 25), f"{int(current_pt):,}", font=font_value_med, fill=(51, 51, 51, 255))
    
    # -- 右侧阵列 (rx=550) --
    # 当前时速
    draw_capsule(draw, rx+30, top_capsule_y, "当前时速", font_capsule_med, bg_color=theme_main)
    speed_prefix = "+" if speed_pt >= 0 else ""
    draw.text((rx+30, top_card_y1 + 25), f"{speed_prefix}{int(speed_pt):,} pt/h", font=font_value_med, fill=theme_main)
    
    # 活动剩余时间
    draw_capsule(draw, rx+30, bot_capsule_y, "剩余时间", font_capsule_med, bg_color=(158, 158, 158, 255))
    draw.text((rx+30, bot_card_y1 + 25), f"{remain_str}", font=font_value_med, fill=(51, 51, 51, 255))
    
    return bg


def draw_bubble_chart_layer(bg, chart_data, theme_main, theme_shadow, theme_main_hex):
    """第四层：元气面积收敛图 (The Bubble Area Chart)"""
    draw = ImageDraw.Draw(bg)
    
    # 图表基座
    draw.rounded_rectangle([(58, 738), (1038, 1808)], radius=25, fill=theme_shadow) # 阴影
    
    # 真实毛玻璃图表基座
    blur_bg = bg.copy().filter(ImageFilter.GaussianBlur(radius=25))
    mask = Image.new('L', bg.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(50, 730), (1030, 1800)], radius=25, fill=255)
    bg.paste(blur_bg, (0,0), mask=mask)
    
    overlay = Image.new('RGBA', bg.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle([(50, 730), (1030, 1800)], radius=25, fill=(255, 255, 255, 180), outline=(255, 255, 255, 200), width=2)
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)
    
    font_capsule = get_font(24, bold=True)
    draw_capsule(draw, 90, 755, "TREND CHART // T1000", font_capsule, theme_main)
    
    # 构建 Matplotlib 图表
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(9, 10)) # 适当缩小图表高度腾出空间
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    ax.tick_params(colors='#9e9e9e', labelsize=12) # 浅灰坐标文字
    for spine in ax.spines.values():
        spine.set_color('none') # 去除边界框
    
    ax.grid(True, color='#e0e0e0', alpha=0.8, linestyle='--')
    
    actual_data = chart_data.get('actual', [])
    predict_data = chart_data.get('predict', [])
    
    all_times = []
    
    if actual_data:
        actual_times = [datetime.fromtimestamp(d['x'] / 1000.0) for d in actual_data]
        actual_pts = [d['y'] for d in actual_data]
        all_times.extend(actual_times)
        
        # 实际线段：只画线由于不需要所有节点都有点
        ax.plot(actual_times, actual_pts, color='#cfd8dc', linewidth=3)
        
        # 使用 numpy 在每日 0 点处插值产生高亮小点
        actual_ms = [d['x'] for d in actual_data]
        if actual_ms:
            min_t = min(actual_times).replace(hour=0, minute=0, second=0, microsecond=0)
            curr = min_t
            if min_t < min(actual_times):
                curr += timedelta(days=1)
            
            day_times = []
            while curr <= max(actual_times):
                day_times.append(curr)
                curr += timedelta(days=1)
                
            if day_times:
                day_ms = [t.timestamp() * 1000 for t in day_times]
                day_pts = np.interp(day_ms, actual_ms, actual_pts)
                ax.plot(day_times, day_pts, color='#cfd8dc', linestyle='None',
                        marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=4)

    if predict_data:
        predict_times = [datetime.fromtimestamp(d['x'] / 1000.0) for d in predict_data]
        predict_pts = [d['y'] for d in predict_data]
        all_times.extend(predict_times)
        
        # 糖果折线区：主连线
        ax.plot(predict_times, predict_pts, color=theme_main_hex, linewidth=3, zorder=5)
        
        predict_ms = [d['x'] for d in predict_data]
        if predict_ms:
            min_t = min(predict_times).replace(hour=0, minute=0, second=0, microsecond=0)
            curr = min_t
            if min_t < min(predict_times):
                curr += timedelta(days=1)
                
            day_times = []
            while curr <= max(predict_times):
                day_times.append(curr)
                curr += timedelta(days=1)
                
            if day_times:
                day_ms = [t.timestamp() * 1000 for t in day_times]
                day_pts = np.interp(day_ms, predict_ms, predict_pts)
                ax.plot(day_times, day_pts, color=theme_main_hex, linestyle='None',
                        marker='o', markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=6)
                
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=0)
    
    # 纵坐标万级简写 (100w 样式)
    def y_fmt(x, pos):
        if x >= 10000:
            return f'{x/10000:g}w'
        return f'{int(x)}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    
    # 限制 y 轴底部不从 0 开始，如果数据过大
    if all_times:
       ax.set_ylim(bottom=max(0, plt.ylim()[0] * 0.9))

    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close(fig)
    
    chart_img = Image.open(buf).convert("RGBA")
    chart_w, chart_h = chart_img.size
    
    # 粘贴到底板上
    bg.paste(chart_img, (90, 800), mask=chart_img)
    
    return bg


def draw_footer(bg, updated_at):
    """第五层：偶像感水印落款 (The Pop Footer)"""
    draw = ImageDraw.Draw(bg)
    font_footer = get_font(24)
    time_str = datetime.fromtimestamp(updated_at / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    text = f"Powered by ZHEnnG, Updated: {time_str}"
    
    bbox = draw.textbbox((0, 0), text, font=font_footer)
    w = bbox[2] - bbox[0]
    
    # 居中对齐
    x = (1080 - w) // 2
    y = 1840
    
    draw.text((x, y), text, font=font_footer, fill=(158, 158, 158, 255))
    return bg


async def create_prediction_poster(data_dict: dict) -> BytesIO:
    """组合全流程生成最终 Pop & Kawaii 海报"""
    event_id = str(data_dict.get('event_id'))
    
    # 0. 拿图
    img_bytes = await fetch_banner_bytes(event_id)
    original_banner = Image.open(BytesIO(img_bytes)).convert("RGBA")
    
    # 1. 引擎计算色彩
    theme_main, theme_light, theme_shadow, theme_main_hex = get_theme_colors(original_banner)
    
    # 2. 第一层底板
    bg = create_pop_background(theme_light)
    
    # 3. 第二层头图
    bg = draw_header_card(bg, original_banner, theme_shadow, theme_main, event_id)
    
    # 4. 第三层数据视窗
    bg = draw_data_board(bg, data_dict, theme_main, theme_shadow)
    
    # 5. 第四层折线图
    chart_data = data_dict.get('chart_data', {})
    bg = draw_bubble_chart_layer(bg, chart_data, theme_main, theme_shadow, theme_main_hex)
    
    # 6. 第五层底栏
    bg = draw_footer(bg, data_dict.get('updated_at', 0))
    
    final_buf = BytesIO()
    bg.save(final_buf, format='PNG')
    final_buf.seek(0)
    
    return final_buf
