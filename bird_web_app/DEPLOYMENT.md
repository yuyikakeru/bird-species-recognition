# 公开发布说明

这个网站有两种发布方式。

## 临时公网地址

临时隧道可以把本地网站公开到一个 HTTPS 地址。这个地址可以被别人访问，但它是临时地址，每次重新启动隧道都可能变化。只有这台电脑同时保持下面两个进程运行时才可用：

- `python .\bird_web_app\server.py --host 0.0.0.0 --port 8000 --warmup`
- `ssh -R 80:127.0.0.1:8000 nokey@localhost.run`

重新生成一个新的临时公网地址：

```powershell
.\bird_web_app\start_public.ps1
```

脚本启动后会打印新的公网地址。

## 永久在线网站

如果需要长期稳定的网址，可以选择下面任意一种方案：

1. 租用云服务器，安装 Python、PyTorch 和项目依赖，把本网站放到 HTTPS 服务后面运行。
2. 使用持久隧道账号，并把自定义域名绑定到本机服务。
3. 使用支持 Python、PyTorch、外网访问和大模型文件的机器学习部署平台。

自定义域名不是必须的；如果想要固定的品牌网址，就需要购买或已经拥有域名。由于这个网站需要 Python 后端加载本地模型，不是纯静态网页，所以长期在线通常也需要云服务器或付费部署平台。

## 运行依赖

先安装项目依赖：

```powershell
pip install -r requirements.txt
```

本网站依赖下面两个本地文件：

```text
ckpt/pipeline/convnext_region_448_final/final/<最高精度运行目录>/convnextv2_tiny_dca_region_final.pt
datasets/CUB_200_2011/CUB_200_2011/classes.txt
```

识别类别只来自本地模型。联网可用时，详细描述会优先从 English Wikipedia 获取；无法联网或查不到时，会自动回退到英文的模型输出说明。
