# 使用 Python 3.10 轻量版作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

ENV PYTHONUNBUFFERED=1
# 设置时区为上海 (CST)
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装基础依赖 (如有需要编译的库可以解开注释，并添加中文字体)
RUN apt-get update && apt-get install -y gcc fonts-wqy-microhei && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到工作目录
COPY . .

# 运行 NoneBot (假设入口文件是 bot.py)
CMD ["python", "bot.py"]