"""
抗裁切水印核心算法 —— 分块 DCT 频域嵌入 / 盲检测。

设计要点
========
* 8x8 像素块 DCT，在 ZigZag 第 11 / 12 位两个中频系数上嵌入 1 bit
  （两个系数的相对大小代表 0/1）。
* 以 T x T 个 8x8 块为一个"瓦片"。每个瓦片中：
    - 约 30% 的块为同步块（由固定 PRBS 决定），嵌入已知同步图案；
    - 其余块为数据块，每块 1 bit，沿瓦片内顺序承载同一份完整比特流。
  整幅图像重复平铺同一个瓦片 => 任意位置、任意尺寸的裁切都能幸存。
* 检测时对 64 种 8x8 像素网格偏移 + 3 种瓦片尺寸 (16/20/24) 做
  同步相关搜索；同步块同时确定网格相位与瓦片相位，无需原图。
* 数据块按瓦片内坐标折叠、对每个比特做跨瓦片多数表决；
  表决差值给出置信度。
* 同步失配 / 数据表决差值过小的块在热力图中标记为疑似失真区域。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------

BLOCK = 8                       # 8x8 像素块
TS = (16, 20, 24)               # 可选瓦片边长（单位：块）
SYNC_FRACTION = 0.30            # 同步块占比
MAGIC = 0x9E5C                  # 帧头魔数（16 bit）
LEN_BITS = 9                    # 载荷长度字段位数
CRC_BITS = 16                   # CRC-16 位数
MAX_TOTAL = 352                 # T=24 时数据块数（=0.70*576≈403，留余量）
MAX_PAYLOAD_BYTES = (MAX_TOTAL - 16 - LEN_BITS - CRC_BITS) // 8  # =38

# ZigZag 扫描第 11、12 位对应的 (u, v)
P1 = (1, 3)
P2 = (2, 2)

# 检测阈值
DETECT_SCORE = 0.30             # 同步相关分阈值
STRANGE_MARGIN = 3.0            # 数据块表决差值（|c1-c2|）低于此值视为可疑
# 置信度饱和对应的最小归一化差值
CONF_SAT = 10.0

# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)。"""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    """正交 DCT-II 基矩阵 D（参考实现；运行时使用 cv2 批量 DCT）。"""
    d = np.empty((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            d[k, i] = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d *= np.sqrt(2.0 / n)
    d[0, :] /= np.sqrt(2.0)
    return d


_D = _dct_matrix()          # 参考正交基（调试 / 对照用）


def _dct2_batch(x2d: np.ndarray, n: int, inverse: bool):
    """对 (n,8,8) 堆叠块做批量 2D DCT/IDCT：行变换后转置做列变换再转回。"""
    fn = cv2.idct if inverse else cv2.dct
    x2d = x2d.reshape(n * BLOCK, BLOCK)
    a = fn(x2d, flags=cv2.DCT_ROWS)                  # (n*8, 8) 行变换
    a = a.reshape(n, BLOCK, BLOCK).transpose(0, 2, 1).reshape(n * BLOCK, BLOCK)
    b = fn(a, flags=cv2.DCT_ROWS)                    # 列变换
    return b.reshape(n, BLOCK, BLOCK).transpose(0, 2, 1)


def block_dct_all(patches: np.ndarray) -> np.ndarray:
    """(N,8,8) 像素块 -> (N,8,8) DCT-II 系数（cv2 批量 2D DCT）。"""
    x = patches.reshape(-1, BLOCK, BLOCK).astype(np.float32)
    return _dct2_batch(x, x.shape[0], False).astype(np.float64) \
        .reshape(patches.shape)


def block_idct_all(coeffs: np.ndarray) -> np.ndarray:
    c = coeffs.reshape(-1, BLOCK, BLOCK).astype(np.float32)
    return _dct2_batch(c, c.shape[0], True).astype(np.float64) \
        .reshape(coeffs.shape)


def pick_t(data_bits: int) -> int:
    """根据总比特数选择能装下它的最小瓦片边长。"""
    for t in TS:
        total = t * t
        n_sync = int(round(SYNC_FRACTION * total))
        if total - n_sync >= data_bits:
            return t
    raise ValueError(f"消息过长：需要 {data_bits} bit，最大 {MAX_TOTAL} bit")


def _rng_for_t(t: int) -> np.random.RandomState:
    # 固定 PRBS：同步图案 / 同步块位置对所有图像相同
    return np.random.RandomState(0x5A17 ^ (t * 7919))


def build_layout(t: int):
    """返回 (sync_mask bool[T,T], sync_pat +-1[T,T])。True=同步块。"""
    total = t * t
    n_sync = int(round(SYNC_FRACTION * total))
    rng = _rng_for_t(t)
    mask = np.zeros(total, dtype=bool)
    mask[:n_sync] = True
    rng.shuffle(mask)
    mask = mask.reshape(t, t)
    pat = rng.choice([-1.0, 1.0], size=total).reshape(t, t)
    return mask, pat


def encode_packet(text: str) -> list[int]:
    payload = text.encode("utf-8")
    if len(payload) == 0:
        raise ValueError("水印文本为空")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"文本过长：{len(payload)} 字节，上限 {MAX_PAYLOAD_BYTES} 字节"
            f"（约 {MAX_PAYLOAD_BYTES // 3} 个汉字）"
        )
    bits: list[int] = []
    for k in range(15, -1, -1):
        bits.append((MAGIC >> k) & 1)
    for k in range(LEN_BITS - 1, -1, -1):
        bits.append((len(payload) >> k) & 1)
    for b in payload:
        for k in range(7, -1, -1):
            bits.append((b >> k) & 1)
    c = crc16(payload)
    for k in range(CRC_BITS - 1, -1, -1):
        bits.append((c >> k) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | (b & 1)
        out.append(v)
    return bytes(out)


# ----------------------------------------------------------------------------
# 图像预处理
# ----------------------------------------------------------------------------

MAX_SIDE = 2048
MIN_SIDE = 128


def prepare_image(pil_img: Image.Image):
    """统一尺寸（长边 <=2048，裁到 8 的倍数），返回 RGB PIL。"""
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    w2, h2 = (w // BLOCK) * BLOCK, (h // BLOCK) * BLOCK
    if min(w2, h2) < MIN_SIDE:
        raise ValueError(f"图像过小：有效尺寸 {w2}x{h2}，最短边需 >= {MIN_SIDE}")
    if (w2, h2) != (w, h):
        img = img.crop((0, 0, w2, h2))
    return img


def _to_blocks(y: np.ndarray) -> np.ndarray:
    """(H,W) 亮度 -> (nb,8,8) 块，行优先。"""
    h, w = y.shape
    bh, bw = h // BLOCK, w // BLOCK
    return (y.reshape(bh, BLOCK, bw, BLOCK)
             .transpose(0, 2, 1, 3)
             .reshape(bh * bw, BLOCK, BLOCK))


def _from_blocks(blocks: np.ndarray, bh: int, bw: int) -> np.ndarray:
    return (blocks.reshape(bh, bw, BLOCK, BLOCK)
                 .transpose(0, 2, 1, 3)
                 .reshape(bh * BLOCK, bw * BLOCK))


def luma_of(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0].astype(np.float64), ycrcb


# ----------------------------------------------------------------------------
# 嵌入
# ----------------------------------------------------------------------------

ALPHA_BY_STRENGTH = {1: 14.0, 2: 20.0, 3: 30.0}


def embed(rgb: np.ndarray, text: str, strength: int = 2) -> dict:
    """在 RGB 图像中嵌入水印，返回结果字典（不修改输入）。"""
    h, w = rgb.shape[:2]
    bh, bw = h // BLOCK, w // BLOCK
    bits = encode_packet(text)
    t = pick_t(len(bits))
    sync_mask, sync_pat = build_layout(t)
    data_positions = np.argwhere(~sync_mask)       # (nd,2) 瓦片内行、列
    nd = len(data_positions)

    y, ycrcb = luma_of(rgb)
    blocks = _to_blocks(y)
    dcts = block_dct_all(blocks)
    c1 = dcts[:, P1[0], P1[1]].copy()
    c2 = dcts[:, P2[0], P2[1]].copy()

    # 每块的目标比特（-1 表示同步块），向量化平铺
    data_tile_idx = np.full((t, t), -1, dtype=np.int32)
    for k, (tr, tc) in enumerate(data_positions):
        data_tile_idx[tr, tc] = k
    rr, cc = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    data_k = data_tile_idx[rr % t, cc % t]
    bit_index = np.where(data_k >= 0, data_k % len(bits), 0)
    data_bits_arr = np.asarray(bits, dtype=np.int8)[bit_index]
    target = np.where(data_k >= 0, data_bits_arr, -1)  # -1 => 同步(-)
    # 同步块符号取 sync_pat
    sync_sign = sync_pat[rr % t, cc % t]
    want_sign = np.where(target < 0, sync_sign,
                         np.where(target == 1, 1.0, -1.0)).reshape(-1)

    # 自适应强度：纹理（AC 能量）越大，系数抬得越高
    ac = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=(1, 2))
                 - blocks.mean(axis=(1, 2)) ** 2)
    alpha = np.clip(
        ALPHA_BY_STRENGTH.get(int(strength), 20.0) * (1.0 + 0.06 * ac),
        0, 60.0,
    )

    # 调整 c1 / c2，使 (c1-c2) 的符号 = want_sign，并留 alpha 余量
    cur = c1 - c2
    goal = want_sign * alpha
    delta = goal - cur
    # 仅当需要时推动；各推动一半
    move = np.where(np.abs(delta) < 1e-9, 0.0, delta)
    c1_new = c1 + 0.5 * move
    c2_new = c2 - 0.5 * move

    dcts[:, P1[0], P1[1]] = c1_new
    dcts[:, P2[0], P2[1]] = c2_new
    blocks_new = block_idct_all(dcts)
    y_new = _from_blocks(blocks_new, bh, bw)

    out = ycrcb.copy()
    out[:, :, 0] = np.clip(np.round(y_new), 0, 255)
    rgb_out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)

    mse = float(np.mean((rgb.astype(np.float64) - rgb_out.astype(np.float64))
                        ** 2))
    psnr = float(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
    return {
        "rgb": rgb_out,
        "tile": t,
        "data_bits": len(bits),
        "capacity": nd,
        "psnr": psnr,
    }


# ----------------------------------------------------------------------------
# 攻击模拟
# ----------------------------------------------------------------------------

def jpeg_recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 10, 100))
    buf = io.BytesIO()
    Image.fromarray(rgb).save(
        buf, format="JPEG", quality=quality, subsampling=2, optimize=True
    )
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def center_crop(rgb: np.ndarray, ratio: float) -> np.ndarray:
    """保留中心 ratio 比例区域（ratio 为边长比例）。"""
    ratio = float(np.clip(ratio, 0.05, 1.0))
    h, w = rgb.shape[:2]
    ch, cw = int(round(h * ratio)), int(round(w * ratio))
    # 裁到 8 的倍数，保证网格整洁（实际检测对任意偏移都鲁棒）
    ch, cw = (ch // BLOCK) * BLOCK, (cw // BLOCK) * BLOCK
    ch, cw = max(ch, BLOCK), max(cw, BLOCK)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw].copy()


# ----------------------------------------------------------------------------
# 检测
# ----------------------------------------------------------------------------

@dataclass
class DetectResult:
    detected: bool = False
    text: str = ""
    confidence: float = 0.0
    score: float = 0.0
    tile: int = 0
    offset: tuple[int, int] = (0, 0)
    n_tiles_votes: int = 0
    bit_conf: list[float] = field(default_factory=list)
    char_conf: list[float] = field(default_factory=list)
    # 块级数据：与原图同网格（bh x bw），值域见 *
    block_margin: np.ndarray | None = None   # 表决差值（强度）
    block_kind: np.ndarray | None = None     # 0 数据 1 同步
    block_strange: np.ndarray | None = None  # True=疑似失真
    width_blocks: int = 0
    height_blocks: int = 0
    length_bytes: int = 0
    crc_ok: bool = False
    error: str = ""


def _fold_phase(soft2d: np.ndarray, bh: int, bw: int, t: int,
                oy: int, ox: int):
    """从块偏移 (oy,ox) 开始取完整瓦片，折叠进 T x T。

    返回 (fold, nh, nx)；fold[a,b] 为可见区内相对相位 (a,b) 上
    所有块软信息之和。
    """
    nh, nx = (bh - oy) // t, (bw - ox) // t
    if nh < 1 or nx < 1:
        return None
    sub = (soft2d[oy:oy + nh * t, ox:ox + nx * t]
           .reshape(nh, t, nx, t)
           .transpose(1, 3, 0, 2)
           .reshape(t, t, nh * nx))
    return sub.sum(axis=2), nh, nx


def _phase_fft(fold: np.ndarray, mask: np.ndarray, pat: np.ndarray):
    """仅在同步模板位置上的掩码循环互相关，一次 FFT 求出全部位移。

    score(py,px) = Σ_{(a,b)∈M} fold[a-py,b-px]·pat[a,b]
                 = (fold ⋆ g)[py,px]，g = pat·mask
    返回 (score 矩阵, py, px)；np.roll(fold,(py,px)) 在同步位置与 pat 对齐。
    """
    g = np.where(mask, pat, 0.0)
    # 循环互相关 R[k] = Σ_i fold[i] g[i+k] = IFFT(conj(F(fold))·F(g))
    dot = np.real(np.fft.ifft2(np.conj(np.fft.fft2(fold))
                               * np.fft.fft2(g)))
    # 位移后掩码覆盖区域内 fold 的能量（同样用一次互相关求）
    f2 = fold ** 2
    en = np.real(np.fft.ifft2(
        np.conj(np.fft.fft2(f2)) * np.fft.fft2(mask.astype(np.float64))))
    denom = np.sqrt(np.maximum(en, 1e-12)) * np.sqrt(
        float(np.sum(pat[mask] ** 2)))
    corr = dot / denom
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    return corr, int(idx[0]), int(idx[1])


def _soft_for_pixels(yf: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """从 float 亮度提取所有块在 P1/P2 两系数上的差值（只做一次 DCT）。"""
    blocks = _to_blocks(yf[:bh * BLOCK, :bw * BLOCK])
    d = block_dct_all(blocks)
    return d[:, P1[0], P1[1]] - d[:, P2[0], P2[1]]


def _block_offset_candidates(soft: np.ndarray, bh: int, bw: int,
                             sy: int = 0, sx: int = 0):
    """固定像素偏移 (sy,sx)，枚举 64 块网格偏移 × 3 瓦片尺寸的候选。"""
    cands = []
    for t in TS:
        mask, pat = build_layout(t)
        for oy in range(BLOCK):
            for ox in range(BLOCK):
                r = _fold_phase(soft, bh, bw, t, oy, ox)
                if r is None:
                    continue
                fold, nh, nx = r
                corr, py, px = _phase_fft(fold, mask, pat)
                cands.append((t, sy, sx, oy, ox,
                              float(corr[py, px]), py, px, nh, nx, soft))
    return cands


def _search_candidates(y_img: np.ndarray, limit: int = 10):
    """两阶段同步搜索。

    1) 快速通道：像素偏移 (0,0) + 64 块网格偏移 × 3 瓦片尺寸（1 次 DCT）。
       覆盖无亚像素错位（原图 / 块对齐裁切）这一最常见情形。
    2) 完整通道：若快速候选 CRC 均失败，枚举 64 种像素亚网格偏移
       （任意原点裁切），每种偏移重新取块 DCT。

    返回按相关分降序的候选：
      [(t, sy8, sx8, oy, ox, score, py, px, nh, nx, soft2d), ...]
    """
    H, W = y_img.shape
    bh0, bw0 = H // BLOCK, W // BLOCK
    soft0 = _soft_for_pixels(y_img, bh0, bw0).reshape(bh0, bw0)
    fast = _block_offset_candidates(soft0, bh0, bw0, 0, 0)
    fast.sort(key=lambda c: -c[5])
    top = fast[:6]
    if any((_decode_candidate(c) or {}).get("ok") for c in top):
        return top

    cands = []
    for sy in range(BLOCK):
        for sx in range(BLOCK):
            if sy == 0 and sx == 0:
                cands.extend(fast)
                continue
            hh, ww = H - sy, W - sx
            bh, bw = hh // BLOCK, ww // BLOCK
            if bh < 16 or bw < 16:
                continue
            sub = np.ascontiguousarray(
                y_img[sy:sy + bh * BLOCK, sx:sx + bw * BLOCK])
            soft = _soft_for_pixels(sub, bh, bw).reshape(bh, bw)
            cands.extend(_block_offset_candidates(soft, bh, bw, sy, sx))
    cands.sort(key=lambda c: -c[5])
    return cands[:limit]


def _validate_stream(hard: np.ndarray, nbits: int):
    """hard: 全数据槽硬判 (nd,)。返回 (ok, info)。"""
    magic = 0
    for b in hard[:16]:
        magic = (magic << 1) | int(b)
    if magic != MAGIC:
        return False, {}
    length = 0
    for b in hard[16:16 + LEN_BITS]:
        length = (length << 1) | int(b)
    total_need = 16 + LEN_BITS + length * 8 + CRC_BITS
    if length == 0 or total_need > nbits:
        return False, {"length": length}
    payload = bits_to_bytes(hard[16 + LEN_BITS:16 + LEN_BITS + length * 8]
                            .tolist())
    crc_need = 0
    base = 16 + LEN_BITS + length * 8
    for b in hard[base:base + CRC_BITS]:
        crc_need = (crc_need << 1) | int(b)
    return crc_need == crc16(payload), {
        "length": length, "payload": payload, "crc": crc_need,
    }


def _decode_candidate(cand):
    """按给定对齐候选做标准折叠 + 解码。

    候选元组：(t, sy8, sx8, oy, ox, score, py, px, nh, nx, soft2d)。
    返回 dict（含 t/score/.../ok/info/hard/norm_margin/dpos/nt 等）。
    """
    t, sy8, sx8, oy, ox, score, py, px, nh, nx, soft = cand
    bh, bw = soft.shape
    sync_mask, sync_pat = build_layout(t)
    grm = (oy - py) % t
    gcm = (ox - px) % t
    Kn = (bh - grm) // t
    Ln = (bw - gcm) // t
    if Kn < 1 or Ln < 1:
        return None
    sub = (soft[grm:grm + Kn * t, gcm:gcm + Ln * t]
           .reshape(Kn, t, Ln, t)
           .transpose(1, 3, 0, 2))                    # (t,t,Kn,Ln)
    samples = sub.reshape(t, t, Kn * Ln)
    sums = samples.sum(axis=2)
    nt = Kn * Ln

    dpos = np.argwhere(~sync_mask)
    nd = len(dpos)
    slot_sums = sums[dpos[:, 0], dpos[:, 1]]
    norm_margin = np.abs(slot_sums) / nt
    hard = (slot_sums > 0).astype(np.int8)
    ok, info = _validate_stream(hard, nd)
    return dict(t=t, score=score, grm=grm, gcm=gcm, Kn=Kn, Ln=Ln,
                sy8=sy8, sx8=sx8, bh=bh, bw=bw,
                sync_mask=sync_mask, sync_pat=sync_pat, dpos=dpos, nd=nd,
                slot_sums=slot_sums, norm_margin=norm_margin, hard=hard,
                sums=sums, nt=nt,
                ok=ok, info=info)


def detect(rgb: np.ndarray) -> DetectResult:
    res = DetectResult()
    h, w = rgb.shape[:2]
    if min(h, w) < 16 * BLOCK:
        res.error = f"图像过小（{w}x{h}），最短边需 >= {16 * BLOCK} 像素"
        return res

    y_img, _ = luma_of(rgb)
    cands = _search_candidates(y_img)
    if not cands:
        res.error = "未找到同步标记：图像可能不含水印或破坏过于严重"
        return res

    # 逐个候选解码：CRC 正确者优先（取其中相关分最高），否则取最高分候选
    pairs = [(cand, _decode_candidate(cand)) for cand in cands]
    pairs = [pc for pc in pairs if pc[1] is not None]
    if not pairs:
        res.error = "定位失败：裁切后图像内不足一个完整瓦片周期"
        return res
    valid = [pc for pc in pairs if pc[1]["ok"]]
    cand, d = (max(valid, key=lambda z: z[1]["score"]) if valid
               else max(pairs, key=lambda z: z[1]["score"]))
    soft = cand[-1]

    t = d["t"]
    sync_mask, sync_pat = d["sync_mask"], d["sync_pat"]
    grm, gcm, nt = d["grm"], d["gcm"], d["nt"]
    sy8, sx8, bh, bw = d["sy8"], d["sx8"], d["bh"], d["bw"]
    hard, norm_margin = d["hard"], d["norm_margin"]
    dpos, nd = d["dpos"], d["nd"]
    score = d["score"]
    ok, info = d["ok"], d["info"]

    res.tile = t
    res.score = score
    # 标准瓦片原点相对输入图像左上角的像素偏移
    res.offset = (sy8 + grm * BLOCK, sx8 + gcm * BLOCK)
    res.width_blocks, res.height_blocks = bw, bh
    res.n_tiles_votes = nt
    res.crc_ok = bool(ok)
    res.detected = bool(ok) and score >= DETECT_SCORE

    # 未通过 CRC 时尽力找一个魔数匹配的旋转
    shift = 0
    stream = hard
    if not ok:
        stream = hard
        for s in range(nd):
            rot = np.roll(hard, -s)
            magic = 0
            for b in rot[:16]:
                magic = (magic << 1) | int(b)
            if magic == MAGIC:
                length = 0
                for b in rot[16:16 + LEN_BITS]:
                    length = (length << 1) | int(b)
                info = {"length": length, "payload": b"", "crc": 0}
                shift = s
                stream = rot
                break
        res.error = ("同步/校验未通过（相关分 %.2f，CRC 校验失败）；"
                     "结果为尽力恢复，仅供参考" % score)

    length = int(info.get("length", 0))
    res.length_bytes = length
    L = min(nd, 16 + LEN_BITS + length * 8 + CRC_BITS)
    m_rot = np.roll(norm_margin, -shift)
    bit_conf = np.clip(m_rot[:L] / CONF_SAT, 0, 1)
    res.bit_conf = [float(x) for x in bit_conf]
    res.confidence = float(np.mean(bit_conf)) if len(bit_conf) else 0.0
    if not ok:
        res.confidence *= 0.5

    if ok:
        res.text = info["payload"].decode("utf-8", errors="replace")
    else:
        raw = stream[16 + LEN_BITS:
                     16 + LEN_BITS + max(length, 1) * 8].tolist()
        res.text = bits_to_bytes(raw).decode("utf-8", errors="replace")

    if ok and length > 0:
        bc = bit_conf[16 + LEN_BITS:16 + LEN_BITS + length * 8]
        if len(bc) == length * 8:
            byte_conf = bc.reshape(length, 8).min(axis=1)
            char_conf, ci = [], 0
            for ch in res.text:
                nby = len(ch.encode("utf-8", errors="replace"))
                seg = byte_conf[ci:ci + nby]
                char_conf.append(float(seg.min()) if len(seg) else 0.0)
                ci += nby
            res.char_conf = char_conf

    # ---- 块级标记（全部块映射到标准模板坐标）----
    block_strange = np.zeros((bh, bw), dtype=bool)
    block_kind = np.zeros((bh, bw), dtype=np.uint8)
    block_margin = np.abs(soft)

    BR, BC = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    cu = (BR - grm) % t
    cv = (BC - gcm) % t
    is_sync = sync_mask[cu, cv]
    block_kind[is_sync] = 1

    # 同步块：仅当该同步槽跨瓦片多数表决后符号仍与模板相反时标红
    sync_vote = d["sums"] * sync_mask
    sync_bad = is_sync & (sync_vote[cu, cv] * sync_pat[cu, cv] < 0)
    block_strange[sync_bad] = True

    data_idx = np.full((t, t), -1, dtype=np.int32)
    for k, (ur, uc) in enumerate(dpos):
        data_idx[ur, uc] = k
    slot_of = data_idx[cu, cv]
    data_sel = slot_of >= 0

    if ok:
        nbits = 16 + LEN_BITS + length * 8 + CRC_BITS
        frame_slot = slot_of % nbits
        in_frame = frame_slot < L
        # 仅标记"多数表决后软差值仍偏弱"的槽对应的全部块
        low = in_frame & (m_rot[frame_slot] < STRANGE_MARGIN)
        block_strange[data_sel & low] = True

    res.block_margin = block_margin
    res.block_kind = block_kind
    res.block_strange = block_strange
    return res


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------

def heatmap_png(rgb: np.ndarray, res: DetectResult) -> bytes:
    """
    生成检测热力图 PNG：
      - 绿色：同步块（半透明）
      - 蓝色：数据块，透明度随软信息强度
      - 红色：疑似失真块（同步失配 / 数据差值过小 / 与多数表决不一致）
    """
    bh, bw = res.height_blocks, res.width_blocks
    # 像素亚网格偏移 = 原点像素偏移去掉整瓦片部分
    sy8 = res.offset[0] % (res.tile * BLOCK) % BLOCK
    sx8 = res.offset[1] % (res.tile * BLOCK) % BLOCK
    layer_small = np.zeros((bh, bw, 3), dtype=np.float64)
    alpha = np.zeros((bh, bw), dtype=np.float64)

    strange = res.block_strange
    kind = res.block_kind
    layer_small[strange] = (230, 60, 60)
    alpha[strange] = 0.40
    sync_sel = (~strange) & (kind == 1)
    layer_small[sync_sel] = (60, 200, 90)
    alpha[sync_sel] = 0.28
    data_sel = (~strange) & (kind == 0)
    layer_small[data_sel] = (70, 130, 240)
    alpha[data_sel] = np.clip(res.block_margin[data_sel] / 30.0, 0, 1) * 0.16

    layer = layer_small.repeat(BLOCK, axis=0).repeat(BLOCK, axis=1)
    alpha_full = alpha.repeat(BLOCK, axis=0).repeat(BLOCK, axis=1)[:, :, None]
    base = rgb[sy8:sy8 + bh * BLOCK, sx8:sx8 + bw * BLOCK]
    out = base.astype(np.float64) * (1 - alpha_full) + layer * alpha_full
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
