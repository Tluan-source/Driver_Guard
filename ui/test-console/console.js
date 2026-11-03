const $ = id => document.getElementById(id);
const state = { models: [], sets: [], ws: null, stream: null, running: false, paused: false, webcam: false, busy: false, timer: null };
const pct = n => n == null ? "—" : `${(100 * n).toFixed(2)}%`;
function status(text, error = false) { $("status").textContent = text; $("status").classList.toggle("error", error); }
async function request(path, body) {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function options(select, rows, value, label) {
  select.replaceChildren(...rows.map(row => { const option = document.createElement("option"); option.value = value(row); option.textContent = label(row); return option; }));
}
function modelMeta() {
  const model = state.models.find(m => m.id === $("model").value);
  $("model-meta").textContent = model ? `${model.mode} · ${model.context_s}s · ${model.class_names.join(" / ")} · inner validation accuracy ${pct(model.validation.accuracy)} · ${model.evaluation_scope || "development"}` : "Chưa có checkpoint camera.";
  $("replay").disabled = !model || !$("clip").value;
  $("synthetic").disabled = $("webcam").disabled = !model;
}
function clips() {
  const set = state.sets.find(s => s.id === $("feature").value);
  options($("clip"), set?.clips || [], c => c.clip_id, c => `${c.clip_id} · ${c.split} · ${c.label_name}`);
  modelMeta();
}
function svg(tag, attrs, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text) node.textContent = text;
  return node;
}
function chart(windows) {
  const root = $("chart"); root.replaceChildren();
  const start = windows[0]?.t || 0, end = windows.at(-1)?.t || start + 1;
  const x = t => 48 + (t - start) / Math.max(end - start, 1) * 570;
  const y = p => 182 - p * 160;
  for (const level of [0, .5, 1]) {
    root.append(svg("line", { x1: 48, x2: 618, y1: y(level), y2: y(level), stroke: "#dce5e8", "stroke-dasharray": level === .5 ? "4" : "0" }));
    root.append(svg("text", { x: 10, y: y(level) + 4, fill: "#5c727b", "font-size": 11 }, String(level)));
  }
  let segment = [];
  function flush() { if (segment.length) root.append(svg("polyline", { points: segment.join(" "), fill: "none", stroke: "#007e83", "stroke-width": 2 })); segment = []; }
  for (const row of windows) {
    if (row.p) {
      segment.push(`${x(row.t)},${y(row.p[1])}`);
      const point = svg("circle", { cx: x(row.t), cy: y(row.p[1]), r: 2, fill: "#007e83" });
      point.append(svg("title", {}, `${row.t}s: ${row.p[1]}`)); root.append(point);
    } else { flush(); root.append(svg("circle", { cx: x(row.t), cy: 188, r: 2, fill: "#a0aeb3" })); }
  }
  flush();
  root.append(svg("text", { x: 48, y: 210, fill: "#5c727b", "font-size": 11 }, `${start}s`));
  root.append(svg("text", { x: 618, y: 210, fill: "#5c727b", "font-size": 11, "text-anchor": "end" }, `${end}s`));
}
$("replay").onclick = async () => {
  $("replay").disabled = true; status("Đang replay feature trên CPU…");
  try {
    const result = await request("/dbg/replay", { model: $("model").value, feature_set: $("feature").value, clip: $("clip").value, stride: 25 });
    $("replay-metrics").replaceChildren(...[["Coverage", pct(result.summary.coverage)], ["Accuracy / clip", pct(result.summary.window_accuracy)], ["Cửa sổ", `${result.summary.n_accepted}/${result.summary.n_windows}`]].map(([title, value]) => {
      const span = document.createElement("span"), strong = document.createElement("strong"); span.textContent = title; strong.textContent = value; span.append(strong); return span;
    }));
    chart(result.windows);
    $("consistency").textContent = `Khớp tiền xử lý: ${result.contract.ok === true ? "có" : result.contract.ok === false ? "không" : "chưa rõ"}. So với CSV Kaggle: ${result.consistency.compared} thời điểm; sai lệch lớn nhất ${result.consistency.max_abs_diff ?? "—"}.`;
    $("replay-detail").textContent = JSON.stringify({ clip: result.clip, summary: result.summary, contract: result.contract, consistency: result.consistency, note: result.note }, null, 2);
    status(`Replay xong trong ${(result.elapsed_ms / 1000).toFixed(2)} giây. ${result.note}`);
  } catch (e) { status(e.message, true); } finally { $("replay").disabled = false; }
};
async function socket() {
  if (state.ws?.readyState === WebSocket.OPEN) return state.ws;
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/dbg/ws`); state.ws = ws;
  ws.onmessage = event => {
    const frame = JSON.parse(event.data);
    if (frame.type === "error") { state.busy = false; status(frame.message, true); return; }
    if (frame.type !== "frame") return;
    state.busy = false;
    const model = frame.model;
    $("live-source").textContent = `${frame.source} · ${(frame.t_ms / 1000).toFixed(1)}s`;
    $("live-label").textContent = model?.label ?? (model?.reason === "current_frame_invalid" ? "Chất lượng tín hiệu thấp" : "Chờ đủ ngữ cảnh");
    $("live-progress").textContent = model ? `${model.history_s}/${model.context_s}s · p(drowsy): ${pct(model.probabilities?.[1])} · ${model.reason || "đã đánh giá"}` : "Chưa gắn checkpoint.";
    $("live-detail").textContent = JSON.stringify({ model, output: frame.layers.output, timing_ms: frame.timing_ms }, null, 2);
  };
  ws.onclose = () => { stop(false); status("Đã ngắt kết nối runtime."); };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = () => reject(new Error("Không kết nối được runtime.")); });
  return ws;
}
function controls(running) { for (const id of ["pause", "reset", "stop"]) $(id).disabled = !running; }
function stop(send = true) {
  if (send && state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ cmd: "stop" }));
  clearInterval(state.timer); state.timer = null;
  state.stream?.getTracks().forEach(track => track.stop()); state.stream = null;
  $("preview").srcObject = null; $("preview").hidden = true;
  state.running = state.webcam = state.paused = state.busy = false;
  $("pause").textContent = "Tạm dừng"; controls(false);
}
async function start(webcam) {
  stop();
  try {
    const ws = await socket();
    if (webcam) {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      $("preview").srcObject = state.stream; $("preview").hidden = false; await $("preview").play();
    }
    ws.send(JSON.stringify({ cmd: "model", id: $("model").value }));
    ws.send(JSON.stringify({ cmd: "start", source: webcam ? "webcam" : "synthetic", scenario: $("scenario").value, speed: Number($("speed").value), fast_warmup: true }));
    state.running = true; state.webcam = webcam; controls(true);
    $("live-label").textContent = "Chờ đủ ngữ cảnh"; $("live-progress").textContent = "0/120s";
    status(webcam ? "Webcam đang xử lý cục bộ. Cần 120 giây tín hiệu trước dự đoán." : "Mô phỏng đang chạy. Đây là kiểm tra runtime.");
    if (webcam) state.timer = setInterval(() => {
      if (state.busy || state.paused || !state.running) return;
      const video = $("preview"), canvas = $("capture");
      if (!video.videoWidth) return;
      canvas.width = 640; canvas.height = Math.round(video.videoHeight * 640 / video.videoWidth);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height); state.busy = true;
      canvas.toBlob(async blob => {
        if (blob && state.running && ws.readyState === WebSocket.OPEN) ws.send(await blob.arrayBuffer());
        else state.busy = false;
      }, "image/jpeg", .8);
    }, 100);
  } catch (e) { stop(); status(e.message, true); }
}
$("synthetic").onclick = () => start(false);
$("webcam").onclick = () => start(true);
$("stop").onclick = () => { stop(); $("live-label").textContent = "Đã dừng"; status("Đã dừng runtime và đóng webcam."); };
$("pause").onclick = () => {
  state.paused = !state.paused; state.ws.send(JSON.stringify({ cmd: state.paused ? "pause" : "resume" }));
  $("pause").textContent = state.paused ? "Tiếp tục" : "Tạm dừng";
};
$("reset").onclick = () => { state.ws.send(JSON.stringify({ cmd: "reset" })); $("live-label").textContent = "Đã xóa lịch sử"; $("live-progress").textContent = "Chờ lại đủ 120 giây."; };
$("model").onchange = () => { modelMeta(); if (state.running) { state.ws.send(JSON.stringify({ cmd: "model", id: $("model").value })); $("live-label").textContent = "Đã đổi model · chờ ngữ cảnh"; } };
$("feature").onchange = clips; $("clip").onchange = modelMeta;
window.addEventListener("beforeunload", () => stop());
try {
  const [catalog, available] = await Promise.all([request("/dbg/models"), request("/dbg/clips")]);
  state.models = catalog.models; state.sets = available.feature_sets;
  options($("model"), state.models, m => m.id, m => `${m.name} · ${m.context_s}s`);
  options($("feature"), state.sets, s => s.id, s => `${s.id} · ${s.clips.length} clip`); clips();
  status(`${state.models.length} checkpoint · ${state.sets.reduce((n, s) => n + s.clips.length, 0)} clip development. Test gốc được khóa.`);
} catch (e) { status(e.message, true); }
