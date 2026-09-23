"use strict";

const $ = (s) => document.querySelector(s);
const state = {
  file: null,
  id: null,
  strength: 2,
  target: "watermarked",
  cache: {},
};

/* ---------------- 工具 ---------------- */
function toast(msg, isErr = false) {
  const t = document.createElement("div");
  t.className = "toast" + (isErr ? " err" : "");
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);
  return data;
}

const imgUrl = (id, kind) =>
  `/api/img/${id}/${kind}?t=${state.cache[kind] || "0"}`;

/* ---------------- ① 嵌入 ---------------- */
const dz = $("#dropzone");
const fileInput = $("#fileInput");

dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

function selectFile(f) {
  if (!f.type.startsWith("image/")) return toast("请选择图像文件", true);
  state.file = f;
  $("#embedBtn").disabled = false;
  dz.querySelector("p b").textContent = f.name;
  dz.querySelector(".muted").textContent =
    `${(f.size / 1024 / 1024).toFixed(2)} MB · 点击可重新选择`;
}

$("#strengthSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  state.strength = parseInt(b.dataset.v, 10);
  $("#strengthSeg").querySelectorAll("button").forEach((x) =>
    x.classList.toggle("active", x === b));
});

const textInput = $("#wmText");
function updateHint() {
  const n = new TextEncoder().encode(textInput.value.trim()).length;
  $("#textHint").textContent =
    n ? `UTF-8 字节数 ${n}（单瓦片容量 38 字节，约 12 个汉字）` : "";
}
textInput.addEventListener("input", updateHint);
updateHint();

$("#embedBtn").addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!state.file) return toast("请先选择图片", true);
  if (!text) return toast("请输入水印文本", true);

  const btn = $("#embedBtn");
  btn.disabled = true; btn.textContent = "嵌入中…";
  try {
    const fd = new FormData();
    fd.append("image", state.file);
    fd.append("text", text);
    fd.append("strength", state.strength);
    const res = await fetch("/api/embed", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "嵌入失败");
    onEmbedded(data);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "嵌入水印";
  }
});

function meta(items) {
  return items.map(([k, v]) =>
    `<div class="meta-item"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join("");
}

function onEmbedded(d) {
  state.id = d.id;
  state.cache = {};
  $("#origImg").src = imgUrl(d.id, "original");
  $("#wmImg").src = imgUrl(d.id, "watermarked");
  $("#dlWm").href = `/api/img/${d.id}/watermarked?download=1`;
  $("#dlOriginal") && 0;
  $("#embedPreview").classList.remove("hidden");
  $("#step2").classList.remove("hidden");
  $("#step3").classList.remove("hidden");
  $("#attackPreview").classList.add("hidden");
  $("#detectResult").classList.add("hidden");

  $("#embedMeta").classList.remove("hidden");
  $("#embedMeta").innerHTML = meta([
    ["输出尺寸", `${d.width} × ${d.height}`],
    ["瓦片边长", `${d.tile_pixels} px（${d.tile_blocks} 块）`],
    ["数据槽 / 用量", `${d.data_slots} / ${d.used_bits} bit`],
    ["水印 PSNR", `${d.psnr_db} dB`],
    ["嵌入耗时", `${d.embed_ms} ms`],
  ]);
  toast("水印嵌入成功，可下载 PNG 或进入攻击模拟");
}

/* ---------------- ② 攻击 ---------------- */
const qRange = $("#qRange"), cRange = $("#cRange");
qRange.addEventListener("input", () => $("#qVal").textContent = qRange.value);
cRange.addEventListener("input", () =>
  $("#cVal").textContent = cRange.value + "%");

async function applyAttack() {
  if (!state.id) return;
  const body = {
    id: state.id,
    jpeg_quality: parseInt(qRange.value, 10),
    crop_ratio: parseInt(cRange.value, 10) / 100,
  };
  try {
    const d = await api("/api/attack", body);
    state.cache.variant = d.cache;
    state.cache.heatmap = d.cache;
    $("#variantImg").src = imgUrl(state.id, "variant");
    $("#dlVariant").href = `/api/img/${state.id}/variant?download=1`;
    $("#attackLabel").textContent =
      `${d.steps} · ${d.width}×${d.height}`;
    $("#attackPreview").classList.remove("hidden");
    // 攻击后自动把检测目标切到受攻击图
    setTarget("variant");
    $("#detectResult").classList.add("hidden");
    toast("受攻击图已生成");
  } catch (e) { toast(e.message, true); }
}
$("#applyAttack").addEventListener("click", applyAttack);

$("#quickCrop50").addEventListener("click", () => {
  cRange.value = 50; $("#cVal").textContent = "50%";
  qRange.value = 100; $("#qVal").textContent = "100";
  applyAttack();
});
$("#quickCropJpeg").addEventListener("click", () => {
  cRange.value = 50; $("#cVal").textContent = "50%";
  qRange.value = 60; $("#qVal").textContent = "60";
  applyAttack();
});

/* ---------------- ③ 检测 ---------------- */
$("#targetSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  setTarget(b.dataset.v);
});
function setTarget(v) {
  state.target = v;
  $("#targetSeg").querySelectorAll("button").forEach((x) =>
    x.classList.toggle("active", x.dataset.v === v));
}

$("#detectBtn").addEventListener("click", async () => {
  if (!state.id) return toast("请先嵌入水印", true);
  if (state.target === "variant" && !state.cache.variant) {
    // 允许：后端会报“尚未生成攻击图像”
  }
  const btn = $("#detectBtn");
  btn.disabled = true; btn.textContent = "检测中…（亚像素搜索可能耗时数秒）";
  try {
    const d = await api("/api/detect", { id: state.id, target: state.target });
    state.cache.heatmap = d.cache;
    renderDetect(d);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "开始检测";
  }
});

function confColor(c) {
  // 0 红 -> 0.5 黄 -> 1 绿
  const r = c >= 0.5 ? Math.round(245 - (c - 0.5) * 2 * 188) : 245;
  const g = c >= 0.5 ? 196 : Math.round(107 + c * 2 * 89);
  return `background:rgba(${r},${g},80,.22);color:rgb(${r},${g + 20},120);`;
}

function renderDetect(d) {
  $("#detectResult").classList.remove("hidden");
  const badge = $("#detBadge");
  if (d.detected) {
    badge.textContent = "✔ 检测到水印（CRC 校验通过）";
    badge.className = "badge ok";
  } else {
    badge.textContent = "✘ 未通过校验";
    badge.className = "badge bad";
  }
  $("#detErr").textContent = d.error || "";

  const rc = $("#recoveredText");
  rc.innerHTML = "";
  const chars = Array.from(d.text || "");
  const confs = d.char_conf && d.char_conf.length
    ? d.char_conf : chars.map(() => (d.crc_ok ? 1 : 0.3));
  chars.forEach((ch, i) => {
    const span = document.createElement("span");
    span.className = "rc-char";
    span.textContent = ch;
    span.title = `置信度 ${(confs[i] ?? 0).toFixed(2)}`;
    span.style.cssText = confColor(confs[i] ?? 0);
    rc.appendChild(span);
  });
  if (!chars.length) rc.textContent = "（未恢复出文本）";

  $("#detectMeta").innerHTML = meta([
    ["整体置信度", `${(d.confidence * 100).toFixed(1)}%`],
    ["同步相关分", d.sync_score.toFixed(3)],
    ["参与表决瓦片", `${d.votes}`],
    ["瓦片尺寸", `${d.tile_pixels} px`],
    ["锁定原点(px)", `(${d.offset_px[0]}, ${d.offset_px[1]})`],
    ["文本字节", `${d.length_bytes}`],
    ["检测耗时", `${d.detect_ms} ms`],
    ["目标", d.target === "variant" ? "受攻击图" : "含水印原图"],
  ]);

  const conf = Math.max(0, Math.min(1, d.confidence));
  $("#confFill").style.width = (conf * 100).toFixed(1) + "%";
  $("#confText").textContent = (conf * 100).toFixed(0) + "%";

  if (d.heatmap) {
    const ht = d.heatmap_target || state.target;
    $("#heatmapBlock").classList.remove("hidden");
    $("#heatImg").src =
      `/api/img/${state.id}/heatmap?target=${ht}&t=${state.cache.heatmap}`;
    $("#dlHeat").href =
      `/api/img/${state.id}/heatmap?target=${ht}&download=1`;
    $("#strangeText").textContent =
      `约 ${(d.strange_ratio * 100).toFixed(1)}% 的块被标记为疑似失真`;
  } else {
    $("#heatmapBlock").classList.add("hidden");
  }
}
