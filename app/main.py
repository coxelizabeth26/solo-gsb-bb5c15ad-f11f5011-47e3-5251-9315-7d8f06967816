"""
抗裁切水印实验台 —— FastAPI 后端。

路由
====
GET  /                     首页（原生 HTML/CSS/JS）
POST /api/embed            上传图片 + 文本 + 强度，执行分块 DCT 嵌入
POST /api/attack           对含水印图做 JPEG 重压 / 中心裁切
POST /api/detect           对目标图做盲检测，返回文本、置信度、失真区域
GET  /api/img/{id}/{kind}  取 PNG（original / watermarked / variant / heatmap）
                           带 ?download=1 时以下载方式返回
"""

from __future__ import annotations

import io
import time
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import watermark as wm

app = FastAPI(title="抗裁切水印实验台", version="1.0.0")

# ----------------------------------------------------------------------------
# 内存存储（实验台用途，不做持久化；简单 LRU 淘汰）
# ----------------------------------------------------------------------------

MAX_ITEMS = 30
STORE: "OrderedDict[str, dict]" = OrderedDict()


def _put(item: dict) -> str:
    item_id = uuid.uuid4().hex[:12]
    STORE[item_id] = item
    STORE.move_to_end(item_id)
    while len(STORE) > MAX_ITEMS:
        STORE.popitem(last=False)
    return item_id


def _get(item_id: str) -> dict:
    if item_id not in STORE:
        raise HTTPException(status_code=404, detail="图像不存在或已过期，请重新上传")
    STORE.move_to_end(item_id)
    return STORE[item_id]


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 请求模型
# ----------------------------------------------------------------------------

class AttackReq(BaseModel):
    id: str
    jpeg_quality: Optional[int] = None      # 10..95；None / >=100 表示不重压
    crop_ratio: Optional[float] = None      # 0.1..0.99；None / 1 表示不裁切


class DetectReq(BaseModel):
    id: str
    target: str = "watermarked"             # watermarked | variant


# ----------------------------------------------------------------------------
# 页面与静态资源
# ----------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/embed")
async def embed(image: UploadFile = File(...),
                text: str = Form(""),
                strength: int = Form(2)):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception:
        raise HTTPException(400, "无法解析图像，请上传 PNG/JPEG/WEBP 等常见格式")

    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "水印文本不能为空")
    if strength not in (1, 2, 3):
        strength = 2

    try:
        pil = wm.prepare_image(pil)
    except ValueError as e:
        raise HTTPException(400, str(e))

    rgb = np.asarray(pil.convert("RGB"))
    t0 = time.time()
    try:
        result = wm.embed(rgb, text, strength=strength)
    except ValueError as e:
        raise HTTPException(400, str(e))
    embed_ms = int((time.time() - t0) * 1000)

    wm_rgb = result["rgb"].astype(np.uint8)
    item_id = _put({
        "original": rgb,
        "watermarked": wm_rgb,
        "variant": None,
        "heatmaps": {},
        "text": text,
    })
    return {
        "id": item_id,
        "text": text,
        "width": int(wm_rgb.shape[1]),
        "height": int(wm_rgb.shape[0]),
        "tile_blocks": int(result["tile"]),
        "tile_pixels": int(result["tile"] * wm.BLOCK),
        "data_slots": int(result["capacity"]),
        "used_bits": int(result["data_bits"]),
        "psnr_db": round(result["psnr"], 2),
        "strength": strength,
        "embed_ms": embed_ms,
        "max_text_bytes": wm.MAX_PAYLOAD_BYTES,
    }


@app.post("/api/attack")
def attack(req: AttackReq):
    item = _get(req.id)
    img = item["watermarked"]
    steps = []

    if req.crop_ratio is not None and req.crop_ratio < 0.999:
        ratio = float(np.clip(req.crop_ratio, 0.1, 0.99))
        img = wm.center_crop(img, ratio)
        steps.append(f"中心裁切 {int(ratio * 100)}%")

    if req.jpeg_quality is not None and req.jpeg_quality < 100:
        q = int(np.clip(req.jpeg_quality, 10, 99))
        img = wm.jpeg_recompress(img, q)
        steps.append(f"JPEG Q{q}")

    item["variant"] = img
    item["heatmaps"].pop("variant", None)
    return {
        "id": req.id,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "steps": " + ".join(steps) if steps else "未做处理",
        "cache": uuid.uuid4().hex[:8],
    }


def _detect_payload(item: dict, target: str):
    if target == "variant":
        if item.get("variant") is None:
            raise HTTPException(400, "尚未生成攻击图像，请先应用攻击")
        img = item["variant"]
    else:
        target = "watermarked"
        img = item["watermarked"]

    t0 = time.time()
    res = wm.detect(img)
    detect_ms = int((time.time() - t0) * 1000)

    heatmap_kind = None
    strange_ratio = 0.0
    if res.block_strange is not None:
        total = int(res.block_strange.size)
        strange_ratio = round(float(res.block_strange.sum()) / max(total, 1), 4)
        if res.detected:
            item["heatmaps"][target] = wm.heatmap_png(img, res)
            heatmap_kind = "heatmap"

    return {
        "detected": bool(res.detected),
        "crc_ok": bool(res.crc_ok),
        "text": res.text,
        "confidence": round(float(res.confidence), 4),
        "char_conf": [round(float(x), 3) for x in res.char_conf],
        "sync_score": round(float(res.score), 4),
        "tile_blocks": int(res.tile),
        "tile_pixels": int(res.tile * wm.BLOCK),
        "votes": int(res.n_tiles_votes),
        "offset_px": [int(res.offset[0]), int(res.offset[1])],
        "length_bytes": int(res.length_bytes),
        "strange_ratio": strange_ratio,
        "detect_ms": detect_ms,
        "error": res.error,
        "target": target,
        "heatmap": heatmap_kind,
        "heatmap_target": target,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "cache": uuid.uuid4().hex[:8],
    }


@app.post("/api/detect")
def detect(req: DetectReq):
    item = _get(req.id)
    return _detect_payload(item, req.target)


@app.get("/api/img/{item_id}/{kind}")
def get_img(item_id: str, kind: str, download: int = 0, target: str = ""):
    item = _get(item_id)
    if kind == "original":
        data, label = item["original"], "original.png"
    elif kind == "watermarked":
        data, label = item["watermarked"], "watermarked.png"
    elif kind == "variant":
        if item.get("variant") is None:
            raise HTTPException(404, "攻击图像不存在")
        data, label = item["variant"], "attacked.png"
    elif kind == "heatmap":
        key = target if target in ("watermarked", "variant") else "watermarked"
        if item["heatmaps"].get(key) is None:
            raise HTTPException(404, "热力图不存在，请先对目标执行检测")
        return Response(content=item["heatmaps"][key], media_type="image/png",
                        headers=_dl_headers("distortion_heatmap.png", download))
    else:
        raise HTTPException(400, f"未知图像类型 {kind}")
    buf = _png_bytes(data)
    return Response(content=buf, media_type="image/png",
                    headers=_dl_headers(label, download))


def _dl_headers(filename: str, download: int) -> dict:
    if download:
        return {"Content-Disposition": f'attachment; filename="{filename}"'}
    return {"Cache-Control": "no-store"}


# 静态资源（挂载在最后，避免覆盖 /api）
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
