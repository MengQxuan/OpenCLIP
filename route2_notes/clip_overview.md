# Route2: OpenCLIP / CLIP 源码拆解笔记

## 目标
- 从源码角度理解 CLIP 的完整前向：tokenize → text encoder → image encoder → projection → normalize → similarity
- 每一步都能用脚本验证张量形状与数值（可复现）

## 本机/服务器环境
- GPU: RTX 4090 24GB
- PyTorch: 2.6.0+cu124
- OpenCLIP: editable install (open_clip_torch)

## 关键入口与文件定位（OpenCLIP）
- create_model_and_transforms: `src/open_clip/factory.py`
- CLIP 主类 / encode_image / encode_text: `src/open_clip/model.py`
- ViT/Transformer 组件: `src/open_clip/transformer.py`
- tokenizer: `src/open_clip/tokenizer.py`
- pretrained 下载与权重映射: `src/open_clip/pretrained.py`

## 公式对齐（CLIP）
- 图像特征：`f_I = normalize(encode_image(I))`
- 文本特征：`f_T = normalize(encode_text(T))`
- 相似度：`logits = exp(logit_scale) * f_I @ f_T^T`
