# 公开发布说明

这个网站已部署到腾讯云服务器。

## 公网地址

当前公网访问地址：

```text
http://81.68.255.235
```

## 部署方式

网站部署在腾讯云服务器上，Python 后端负责加载本地模型权重并执行推理，Nginx 负责对外提供 HTTP 访问并反向代理到后端服务。

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
