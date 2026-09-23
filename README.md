# 抗裁切水印实验台（Crop-Resistant DCT Watermark Bench）

基于 **Python FastAPI + Pillow / OpenCV + 原生 HTML/CSS/JS** 的频域盲水印实验应用。
把短文本以**分块 DCT**方式嵌入图像亮度通道，水印与**同步标记**以"瓦片"形式
**重复平铺**到整幅图像；即使经过 **JPEG 重压缩**和**中心裁切**，检测端也能：

- 🔍 自动枚举像素 / 块网格偏移，定位同步块（无需原图）；
- 🗳️ 对跨瓦片的重复比特做**多数表决**恢复文本；
- 📊 给出整体置信度、逐字符置信度、同步相关分、表决瓦片数；
- 🗺️ 输出**疑似失真区域热力图**（同步失配 / 表决差值过弱 / 与多数不一致的块）；
- 🖼️ 提供原图 / 含水印图预览，并支持 **PNG 下载**。

---

## 一、一键启动（Docker Compose）

```bash
# 构建并后台启动
docker compose up -d --build

# 查看日志
docker compose logs -f
```

启动成功后浏览器访问：

> **http://localhost:8000**

停止：

```bash
docker compose down
```

### 可选配置（环境变量 / `.env`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST_PORT` | `8000` | 宿主机映射端口（如想用 80 端口：`HOST_PORT=80 docker compose up -d`） |
| `WORKERS` | `1` | uvicorn worker 数；检测为 CPU 密集型，多并发检测时可调大 |

也可直接用 Docker：

```bash
docker build -t crop-resistant-watermark .
docker run --rm -p 8000:8000 crop-resistant-watermark
```

---

## 二、本地运行（无 Docker）

需要 Python 3.11（3.10+ 亦可）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问 **http://localhost:8000**。

---

## 三、使用流程

1. **上传图片并嵌入**：选择图片（长边 ≤ 2048，最短边 ≥ 128），输入短文本水印，
   选择嵌入强度（弱 / 中 / 强），点击"嵌入水印"。可立即下载无损 PNG。
2. **模拟传输损伤**：拖动滑块设置 JPEG 质量（10–100）与中心裁切保留比例
   （15%–100%），生成受攻击图（可叠加）。
3. **盲检测**：选择检测"含水印原图"或"受攻击图"，点击"开始检测"。
   查看恢复文本（逐字符按置信度着色）、置信度条与失真热力图，并可下载热力图 PNG。

> 提示：检测端会自动搜索 64 种像素网格偏移。**原图 / 块对齐图像检测约 0.2 秒**；
> 任意原点的非对齐裁切最坏需要数秒（像素亚网格穷举），属正常现象。

---

## 四、算法原理

```
嵌入比特流 = 魔数(16b) + 载荷长度(9b) + UTF-8 文本 + CRC-16(16b)
```

1. **分块 DCT**：图像转 YCrCb，仅在 **Y（亮度）** 通道上把每个 8×8 块做二维 DCT。
   比较 ZigZag 扫描中第 11、12 位两个中频系数 `(u,v)=(1,3)` 与 `(2,2)`：
   令 `c(1,3) − c(2,2)` 的符号为正/负，承载 1 bit。中频系数对 JPEG 压缩、
   轻度几何处理鲁棒，且视觉上几乎不可见（默认 PSNR ≈ 40 dB）。
2. **瓦片平铺（抗裁切核心）**：以 `T×T` 个块（T∈{16,20,24}，由文本长度选择，
   对应 128/160/192 像素）为一个**瓦片**：
   - 约 **30% 的块为同步块**（位置与 ±1 图案由固定 PRBS 决定），用于定位；
   - 其余为**数据块**，按行优先顺序承载同一份完整比特流；
   - 同一个瓦片在整幅图像上**周期平铺**，因此裁掉任意边缘都能在剩余区域找到
     完整瓦片。
3. **盲检测 / 同步**：枚举 **64 种像素亚网格偏移**（应对非 8 倍数裁切原点）×
   3 种瓦片尺寸 × 块偏移，对折叠后的 `T×T` 相关面用 **FFT 掩码循环互相关**
   一次性求出瓦片相位；候选对齐再用比特流的 **魔数 + CRC-16** 校验选定，
   从而区分真锁与假锁。
4. **多数表决**：每个数据槽在可见区域内出现的所有瓦片实例上累加软差值
   （`Σ(c1−c2)`），用累加符号判决比特。瓦片越多，表决余量越大、置信度越高。
5. **置信度与失真定位**：
   - 比特置信度 = 该槽**平均每块软差值**（饱和归一化）；逐字符取其字节内最小值；
   - 同步块符号与模板相反、数据槽帧内余量过小、或单块软信息与多数表决相反时，
     在热力图中标记为红色"疑似失真"。

### 文本容量

单个瓦片数据槽有限：**最长约 38 字节 UTF-8（约 12 个汉字 / 38 个 ASCII）**。
文本越短，每个比特在瓦片内重复次数越多、越抗损伤。

### 实测鲁棒性（默认中等强度，1024×768 合成自然图）

| 攻击 | 结果 |
| --- | --- |
| 无攻击 | 完美恢复，PSNR ≈ 40 dB |
| JPEG Q90 / Q60 / Q40 | CRC 通过，置信度 ~100% |
| 中心裁切保留 75% / 50% | CRC 通过，置信度 ~100% |
| 中心裁切保留 35%（含非 8 倍数原点） | CRC 通过，置信度 ≥ 98% |
| 裁切 40–55% + JPEG Q45–55（叠加） | CRC 通过，置信度 ≥ 98% |

> 物理极限：当裁切区域只剩 **1 个瓦片**、文本又接近容量上限、同时伴随强 JPEG
> （如 Q45 以下）时，个别比特可能翻转。此时 CRC 会失败、整体置信度下调并明确
> 提示"结果为尽力恢复，仅供参考"，而不会给出错误却自称可信的结果。

---

## 五、HTTP API

服务自带交互式文档：**http://localhost:8000/docs**（Swagger UI）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/embed` | multipart：`image`、`text`、`strength`(1/2/3)，返回任务 id 与嵌入指标 |
| POST | `/api/attack` | JSON：`id`、`jpeg_quality`、`crop_ratio`，生成受攻击图 |
| POST | `/api/detect` | JSON：`id`、`target`(`watermarked`/`variant`)，返回文本/置信度/指标 |
| GET | `/api/img/{id}/{kind}` | `kind`：`original`/`watermarked`/`variant`/`heatmap`；加 `?download=1` 下载 |
| GET | `/api/health` | 健康检查 |

图像保存在**进程内存**（LRU 30 项，重启即失效），仅适合实验台单机用途。

---

## 六、项目结构

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── app
    ├── __init__.py
    ├── watermark.py      # 核心：DCT 嵌入 / 攻击模拟 / 同步搜索 / 表决检测 / 热力图
    ├── main.py           # FastAPI 路由与内存存储
    └── static
        ├── index.html    # 单页界面（原生）
        ├── style.css
        └── app.js
```

核心算法与 Web 框架解耦，`app/watermark.py` 可单独使用：

```python
import numpy as np
from PIL import Image
from app import watermark as wm

img = np.asarray(wm.prepare_image(Image.open("a.png")))
out = wm.embed(img, "机密-2026", strength=2)
Image.fromarray(out["rgb"]).save("wm.png")

attacked = wm.jpeg_recompress(wm.center_crop(out["rgb"], 0.5), 60)
res = wm.detect(attacked)
print(res.detected, res.text, res.confidence)
open("heat.png", "wb").write(wm.heatmap_png(attacked, res))
```

---

## 七、参数调节（`app/watermark.py` 顶部常量）

| 常量 | 含义 |
| --- | --- |
| `ALPHA_BY_STRENGTH` | 三档嵌入基础强度（越大越抗损、但越可见） |
| `TS` | 瓦片边长候选（块） |
| `SYNC_FRACTION` | 同步块占比（默认 0.30） |
| `DETECT_SCORE` | 判定检测成功的同步相关分阈值 |
| `STRANGE_MARGIN` / `CONF_SAT` | 失真判定与置信度归一化的软差值阈值 |
