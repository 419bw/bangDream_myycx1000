from pydantic import BaseModel

class Config(BaseModel):
    # YCX 插件配置 API 地址
    ycx_api_url: str = "http://lightgm_server:5000"