# 鸟类图像识别网站

这是一个独立的网站目录，会加载 `convnext_region_448_final` 实验中测试准确率最高的 ConvNeXtV2 DCA-Region 448 权重，用于本地 Web 演示和答辩展示。

- 模型：`convnextv2_tiny_dca_region`
- 测试 Top-1：`90.5247%`
- 权重文件：`ckpt/pipeline/convnext_region_448_final/final/<最高精度运行目录>/convnextv2_tiny_dca_region_final.pt`

从项目根目录启动本地服务：

```powershell
python .\bird_web_app\server.py --host 0.0.0.0 --port 8000 --warmup
```

本机访问：

```text
http://127.0.0.1:8000
```

同一局域网设备可以访问服务启动时打印的局域网地址，例如：

```text
http://192.168.1.3:8000
```

生成临时公网 HTTPS 地址：

```powershell
.\bird_web_app\start_public.ps1
```

鸟类类别只由当前本地模型预测，页面展示英文类别名。联网时，网站会优先从 English Wikipedia 获取鸟类详细描述；如果联网请求失败，会回退到基于模型输出的英文说明。
