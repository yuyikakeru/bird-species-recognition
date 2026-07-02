# 鸟类图像识别网站

这是一个独立的网站目录，会加载 `convnext_region_448_final` 实验中测试准确率最高的 ConvNeXtV2 DCA-Region 448 权重，用于网页端鸟类图片识别和答辩展示。网站已部署到腾讯云服务器。

- 模型：`convnextv2_tiny_dca_region`
- 测试 Top-1：`90.5247%`
- 权重文件：`ckpt/pipeline/convnext_region_448_final/final/<最高精度运行目录>/convnextv2_tiny_dca_region_final.pt`

公网访问地址：

```text
http://81.68.255.235
```

本地测试时，可从项目根目录启动服务：

```powershell
python .\bird_web_app\server.py --host 0.0.0.0 --port 8000 --warmup
```

本机访问：

```text
http://127.0.0.1:8000
```

鸟类类别只由当前本地模型预测，页面展示英文类别名。联网时，网站会优先从 English Wikipedia 获取鸟类详细描述；如果联网请求失败，会回退到基于模型输出的英文说明。
