const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const layers = { map: null, global: null, local: null };
let status = {};
let paths = { global: [], local: [] };
let robotPose = null;
let goalPose = null;
let initialCandidate = null;
let liveScan = { points: [], count: 0, nearest: null, rangeMax: 10, receivedAt: 0 };
const visibility = {
  grid: true, map: true, global: true, local: true, scan: true,
  globalPath: true, localPath: true, robot: true, cart: true, footprint: true, radar: true,
};
const camera = { zoom: 1, panX: 0, panY: 0 };
let mode = 'view';
let dragStart = null;
let dragCurrent = null;
let panDrag = null;
let rosSocket = null;
let speedDirty = false;
let speedRequestBusy = false;
let speedRequestError = '';
let preparationPending = null;
let preparationError = '';
let modelSetPending = null;
let modelSetError = '';
let cartBusy = false;
let cartError = '';
let cartPending = null;
let cartPoseDirty = false;
let lifecyclePending = null;
let mapBusy = false;
let poseBusy = false;
let motionBusy = false;
let lastStatusAt = -Infinity;
let lastStatusStamp = -Infinity;
let statusPolling = false;

const $ = id => document.getElementById(id);
const api = async path => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, { cache: 'no-store', signal: controller.signal });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || '请求失败');
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('请求超时，请核对当前状态后重试');
    throw error;
  } finally { clearTimeout(timer); }
};
function acceptStatus(next) {
  const stamp = Number(next.sent_monotonic);
  if (Number.isFinite(stamp) && stamp < lastStatusStamp) return;
  if (Number.isFinite(stamp)) lastStatusStamp = stamp;
  status = next;
  lastStatusAt = performance.now();
  updateUi();
  draw();
}
function statusFresh() { return performance.now() - lastStatusAt < 2500; }
async function refreshStatus() { acceptStatus((await api('/api/status')).status); }

function connectRosbridge() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  rosSocket = new WebSocket(`${scheme}://${location.hostname}:9090`);
  rosSocket.onopen = () => {
    $('connection').textContent = 'ROSBRIDGE 在线';
    $('connection').className = 'pill online';
    const topics = [
      ['/web/map_bitmap', 'std_msgs/msg/String'],
      ['/web/global_costmap_bitmap', 'std_msgs/msg/String'],
      ['/web/local_costmap_bitmap', 'std_msgs/msg/String'],
      ['/web/nav_status', 'std_msgs/msg/String'],
      ['/web/global_path', 'std_msgs/msg/String'],
      ['/web/local_path', 'std_msgs/msg/String'],
      ['/amcl_pose', 'geometry_msgs/msg/PoseWithCovarianceStamped'],
      ['/scan', 'sensor_msgs/msg/LaserScan'],
    ];
    topics.forEach(([topic, type], i) => rosSocket.send(JSON.stringify({
      op: 'subscribe', id: `sub_${i}`, topic, type,
      throttle_rate: topic === '/scan' || topic === '/web/local_costmap_bitmap'
        ? 100 : (topic.includes('costmap') ? 250 : 0),
    })));
  };
  rosSocket.onclose = () => {
    $('connection').textContent = 'ROSBRIDGE 离线';
    $('connection').className = 'pill offline';
    setTimeout(connectRosbridge, 1500);
  };
  rosSocket.onerror = () => rosSocket.close();
  rosSocket.onmessage = async event => {
    let packet;
    try { packet = JSON.parse(event.data); } catch (_) { return; }
    if (packet.op !== 'publish') return;
    const topic = packet.topic;
    if (topic.startsWith('/web/') && topic.endsWith('bitmap')) {
      try {
        const meta = JSON.parse(packet.msg.data);
        const bytes = Uint8Array.from(atob(meta.png), c => c.charCodeAt(0));
        meta.bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
        layers[meta.layer] = meta;
        draw();
      } catch (error) { console.warn(error); }
    } else if (topic === '/web/nav_status') {
      try { acceptStatus(JSON.parse(packet.msg.data)); } catch (error) { console.warn(error); }
    } else if (topic === '/web/global_path' || topic === '/web/local_path') {
      try {
        const path = JSON.parse(packet.msg.data);
        paths[path.layer] = path.points || [];
        draw();
      } catch (error) { console.warn(error); }
    } else if (topic === '/amcl_pose') {
      robotPose = poseFromRos(packet.msg.pose.pose);
      draw();
    } else if (topic === '/scan') {
      updateScan(packet.msg);
    }
  };
}

function updateScan(message) {
  const angleMin = Number(message.angle_min);
  const angleIncrement = Number(message.angle_increment);
  const rangeMin = Number(message.range_min) || 0;
  const configuredMax = Number(message.range_max) || 10;
  const rangeMax = Math.min(configuredMax, 10);
  const points = [];
  let nearest = Infinity;
  (message.ranges || []).forEach((raw, index) => {
    const range = Number(raw);
    if (!Number.isFinite(range) || range < rangeMin || range > rangeMax) return;
    const angle = angleMin + index * angleIncrement;
    points.push([range * Math.cos(angle), range * Math.sin(angle), range]);
    nearest = Math.min(nearest, range);
  });
  liveScan = {
    points,
    count: points.length,
    bins: (message.ranges || []).length,
    nearest: Number.isFinite(nearest) ? nearest : null,
    rangeMax,
    receivedAt: performance.now(),
  };
  $('scanStats').textContent = points.length
    ? `避障雷达：${points.length}/${liveScan.bins} 命中 · 最近 ${nearest.toFixed(2)} m · 最大 ${rangeMax.toFixed(1)} m`
    : '避障雷达在线，4 m 内没有障碍命中';
  draw();
}

function poseFromRos(pose) {
  const q = pose.orientation;
  return {
    x: pose.position.x, y: pose.position.y,
    yaw: Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
  };
}

function viewTransform() {
  const base = layers.map;
  if (!base) return null;
  const margin = 30;
  const worldW = base.width * base.resolution;
  const worldH = base.height * base.resolution;
  const fitScale = Math.min((canvas.width - 2 * margin) / worldW, (canvas.height - 2 * margin) / worldH);
  const scale = fitScale * camera.zoom;
  const left = (canvas.width - worldW * scale) / 2 + camera.panX;
  const top = (canvas.height - worldH * scale) / 2 + camera.panY;
  return { base, scale, left, top, maxY: base.origin[1] + worldH };
}

function worldToScreen(x, y, transform = viewTransform()) {
  if (!transform) return null;
  return [
    transform.left + (x - transform.base.origin[0]) * transform.scale,
    transform.top + (transform.maxY - y) * transform.scale,
  ];
}

function screenToWorld(x, y) {
  const t = viewTransform();
  if (!t) return null;
  return [t.base.origin[0] + (x - t.left) / t.scale, t.maxY - (y - t.top) / t.scale];
}

function drawLayer(layer, alpha = 1) {
  const t = viewTransform();
  if (!t || !layer || !layer.bitmap) return;
  const yaw = Number(layer.yaw) || 0;
  const cosine = Math.cos(yaw), sine = Math.sin(yaw);
  const heightMeters = layer.height * layer.resolution;
  const topLeftX = layer.origin[0] - sine * heightMeters;
  const topLeftY = layer.origin[1] + cosine * heightMeters;
  const x = t.left + (topLeftX - t.base.origin[0]) * t.scale;
  const y = t.top + (t.maxY - topLeftY) * t.scale;
  const pixelScale = layer.resolution * t.scale;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = false;
  ctx.transform(
    cosine * pixelScale,
    -sine * pixelScale,
    sine * pixelScale,
    cosine * pixelScale,
    x,
    y,
  );
  ctx.drawImage(layer.bitmap, 0, 0);
  ctx.restore();
}

function drawGrid(dpr) {
  const t = viewTransform();
  if (!t) return;
  const raw = 55 * dpr / t.scale;
  const power = 10 ** Math.floor(Math.log10(raw));
  const ratio = raw / power;
  const spacing = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
  const minX = t.base.origin[0], maxX = minX + t.base.width * t.base.resolution;
  const minY = t.base.origin[1], maxY = minY + t.base.height * t.base.resolution;
  const bottomLeft = worldToScreen(minX, minY, t), topRight = worldToScreen(maxX, maxY, t);
  ctx.save();
  ctx.beginPath();
  ctx.rect(bottomLeft[0], topRight[1], topRight[0] - bottomLeft[0], bottomLeft[1] - topRight[1]);
  ctx.clip();
  ctx.strokeStyle = '#60738648'; ctx.lineWidth = dpr;
  ctx.fillStyle = '#91a4b5'; ctx.font = `${9 * dpr}px sans-serif`; ctx.textAlign = 'left';
  for (let x = Math.ceil(minX / spacing) * spacing; x <= maxX; x += spacing) {
    const p0 = worldToScreen(x, minY, t), p1 = worldToScreen(x, maxY, t);
    ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
    ctx.fillText(`${x.toFixed(spacing < 1 ? 1 : 0)} m`, p1[0] + 2 * dpr, p1[1] + 11 * dpr);
  }
  for (let y = Math.ceil(minY / spacing) * spacing; y <= maxY; y += spacing) {
    const p0 = worldToScreen(minX, y, t), p1 = worldToScreen(maxX, y, t);
    ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
    ctx.fillText(`${y.toFixed(spacing < 1 ? 1 : 0)}`, p0[0] + 3 * dpr, p0[1] - 3 * dpr);
  }
  ctx.restore();
}

function drawPath(points, color, width, dash = []) {
  if (points.length < 2) return;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.setLineDash(dash); ctx.beginPath();
  points.forEach(([x, y], i) => { const p = worldToScreen(x, y); if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
  ctx.stroke(); ctx.restore();
}

function candidatePose() {
  if (mode === 'initial' && dragStart && dragCurrent) {
    return {
      x: dragStart[0], y: dragStart[1],
      yaw: Math.atan2(dragCurrent[1] - dragStart[1], dragCurrent[0] - dragStart[0]),
    };
  }
  return robotPose || initialCandidate;
}

function drawScanOverlay(pose, dpr) {
  if (!pose || !liveScan.points.length) return;
  const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  ctx.save();
  ctx.fillStyle = '#24e6ff';
  ctx.shadowColor = '#1cc9e8';
  ctx.shadowBlur = 3 * dpr;
  const radius = Math.max(1.3, 1.7 * dpr);
  liveScan.points.forEach(([x, y]) => {
    const world = worldToScreen(pose.x + c * x - s * y, pose.y + s * x + c * y);
    if (!world) return;
    ctx.beginPath(); ctx.arc(world[0], world[1], radius, 0, Math.PI * 2); ctx.fill();
  });
  ctx.restore();
}

function drawRadarInset(dpr) {
  const size = Math.min(230 * dpr, canvas.width * .38, canvas.height * .42);
  if (size < 100) return;
  const x0 = 12 * dpr, y0 = 58 * dpr;
  const cx = x0 + size / 2, cy = y0 + size / 2 + 7 * dpr;
  const radiusPx = size / 2 - 18 * dpr;
  const rangeMax = Math.max(2, liveScan.rangeMax || 10);
  const scale = radiusPx / rangeMax;
  ctx.save();
  ctx.fillStyle = '#071019ee'; ctx.strokeStyle = '#405267'; ctx.lineWidth = dpr;
  ctx.fillRect(x0, y0, size, size); ctx.strokeRect(x0, y0, size, size);
  ctx.beginPath(); ctx.rect(x0, y0, size, size); ctx.clip();
  ctx.strokeStyle = '#294052'; ctx.lineWidth = dpr;
  [0.25, 0.5, 0.75, 1].forEach(fraction => {
    ctx.beginPath(); ctx.arc(cx, cy, radiusPx * fraction, 0, Math.PI * 2); ctx.stroke();
  });
  ctx.beginPath(); ctx.moveTo(cx - radiusPx, cy); ctx.lineTo(cx + radiusPx, cy);
  ctx.moveTo(cx, cy - radiusPx); ctx.lineTo(cx, cy + radiusPx); ctx.stroke();
  ctx.fillStyle = '#24e6ff';
  liveScan.points.forEach(([x, y]) => {
    ctx.beginPath(); ctx.arc(cx + x * scale, cy - y * scale, Math.max(1.2, 1.5 * dpr), 0, Math.PI * 2); ctx.fill();
  });
  ctx.fillStyle = '#ffe35b';
  ctx.beginPath(); ctx.arc(cx, cy, 4 * dpr, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#ffe35b'; ctx.lineWidth = 2 * dpr;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + 15 * dpr, cy); ctx.stroke();
  ctx.fillStyle = '#c9d7e4'; ctx.font = `${10 * dpr}px sans-serif`; ctx.textAlign = 'left';
  ctx.fillText('实时雷达 · 前方 →', x0 + 9 * dpr, y0 + 15 * dpr);
  ctx.textAlign = 'right';
  const age = liveScan.receivedAt ? Math.max(0, (performance.now() - liveScan.receivedAt) / 1000) : null;
  ctx.fillStyle = age !== null && age < 1.5 ? '#7be6a2' : '#ff8d95';
  ctx.fillText(age === null ? '等待 /scan' : `${age.toFixed(1)} s`, x0 + size - 9 * dpr, y0 + 15 * dpr);
  ctx.restore();
}

function drawArrow(pose, color, radius = 16, label = '') {
  if (!pose) return;
  const p = worldToScreen(pose.x, pose.y); if (!p) return;
  const dx = Math.cos(pose.yaw) * radius, dy = -Math.sin(pose.yaw) * radius;
  ctx.save(); ctx.translate(...p); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(0, 0, 6, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(dx, dy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(dx, dy); ctx.lineTo(dx - Math.cos(pose.yaw - .55) * 8, dy + Math.sin(pose.yaw - .55) * 8); ctx.lineTo(dx - Math.cos(pose.yaw + .55) * 8, dy + Math.sin(pose.yaw + .55) * 8); ctx.closePath(); ctx.fill();
  if (label) {
    ctx.font = `${11 * Math.max(1, window.devicePixelRatio || 1)}px sans-serif`;
    ctx.textAlign = 'left'; ctx.fillText(label, 9, -9);
  }
  ctx.restore();
}

function draw() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.round(rect.width * dpr), height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!layers.map) {
    ctx.fillStyle = '#718397'; ctx.font = `${14 * dpr}px sans-serif`; ctx.textAlign = 'center';
    ctx.fillText('选择并处理 PCD 后显示地图', canvas.width / 2, canvas.height / 2);
    if (visibility.radar) drawRadarInset(dpr); return;
  }
  if (visibility.map) drawLayer(layers.map, 1);
  if (visibility.grid) drawGrid(dpr);
  if (visibility.global) drawLayer(layers.global, .42);
  if (visibility.local) drawLayer(layers.local, .78);
  if (visibility.scan) drawScanOverlay(candidatePose(), dpr);
  if (visibility.globalPath) drawPath(paths.global, '#28c8f2', 3 * dpr, [8 * dpr, 5 * dpr]);
  if (visibility.localPath) drawPath(paths.local, '#56f39a', 4 * dpr);
  drawCart(dpr);
  if (visibility.robot) drawArrow(robotPose, '#ffe35b', 20 * dpr, 'G1');
  drawArrow(goalPose, '#42ddff', 18 * dpr, 'Goal');
  if (dragStart && dragCurrent) {
    const a = worldToScreen(...dragStart), b = worldToScreen(...dragCurrent);
    ctx.save(); ctx.strokeStyle = mode === 'initial' ? '#ffe35b' : '#42ddff'; ctx.lineWidth = 3 * dpr; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.restore();
  }
  if (visibility.radar) drawRadarInset(dpr);
  $('zoomReadout').textContent = `Zoom: ${Math.round(camera.zoom * 100)}%`;
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
}
canvas.addEventListener('pointerdown', event => {
  if (!layers.map || poseBusy) return;
  canvas.setPointerCapture(event.pointerId);
  if (mode === 'view') {
    panDrag = { point: pointerPosition(event), panX: camera.panX, panY: camera.panY };
    canvas.style.cursor = 'grabbing';
    return;
  }
  dragStart = screenToWorld(...pointerPosition(event)); dragCurrent = dragStart; draw();
});
canvas.addEventListener('pointermove', event => {
  if (panDrag) {
    const point = pointerPosition(event);
    camera.panX = panDrag.panX + point[0] - panDrag.point[0];
    camera.panY = panDrag.panY + point[1] - panDrag.point[1];
    draw(); return;
  }
  if (!dragStart) return; dragCurrent = screenToWorld(...pointerPosition(event)); draw();
});
canvas.addEventListener('pointerup', async event => {
  if (panDrag) {
    panDrag = null; canvas.style.cursor = 'grab'; return;
  }
  if (!dragStart) return;
  dragCurrent = screenToWorld(...pointerPosition(event));
  if (!dragCurrent) {
    dragStart = dragCurrent = null;
    setMode('view');
    draw();
    return;
  }
  // Keep an immutable snapshot across the request. The backend publishes
  // navigation_active before /api/goal returns, and updateUi() is then allowed
  // to clear the global drag state while this handler is awaiting the response.
  const targetPose = {
    x: dragStart[0],
    y: dragStart[1],
    yaw: Math.atan2(dragCurrent[1] - dragStart[1], dragCurrent[0] - dragStart[0]),
  };
  const params = new URLSearchParams(targetPose);
  const requestedMode = mode;
  poseBusy = true;
  updateUi();
  try {
    if (requestedMode === 'initial') {
      if (status.localized && !window.confirm('当前 AMCL 定位仍然有效。重新设置初始位姿会关闭运动门并重新定位，确定继续吗？')) {
        setMode('view');
        dragStart = dragCurrent = null;
        draw();
        return;
      }
      if (status.localized) params.set('replace', 'true');
      await api(`/api/initial_pose?${params}`);
      initialCandidate = targetPose;
    } else if (requestedMode === 'goal') {
      paths = { global: [], local: [] };
      draw();
      await api(`/api/goal?${params}`);
      goalPose = targetPose;
    }
    setMode('view');
    await refreshStatus();
  } catch (error) { alert(error.message); }
  finally { poseBusy = false; updateUi(); }
  dragStart = dragCurrent = null; draw();
});
canvas.addEventListener('pointercancel', () => {
  panDrag = null; dragStart = dragCurrent = null; setMode('view'); draw();
});
canvas.addEventListener('wheel', event => {
  if (!layers.map) return;
  event.preventDefault();
  const pointer = pointerPosition(event);
  const world = screenToWorld(...pointer);
  camera.zoom = Math.max(.5, Math.min(12, camera.zoom * Math.exp(-event.deltaY * .0015)));
  const moved = worldToScreen(...world);
  camera.panX += pointer[0] - moved[0]; camera.panY += pointer[1] - moved[1];
  draw();
}, { passive: false });

function setMode(next) {
  mode = next;
  $('modeBadge').textContent = next === 'initial'
    ? '2D 初始位姿：拖拽位置与朝向'
    : next === 'goal'
      ? 'Nav2 Goal：拖拽位置与朝向'
      : '平移 / 缩放模式';
  canvas.style.cursor = next === 'view' ? 'grab' : 'crosshair';
  $('panMode').classList.toggle('active', next === 'view');
  $('toolbarInitial').classList.toggle('active', next === 'initial');
  $('toolbarGoal').classList.toggle('active', next === 'goal');
}

function resetCamera() {
  camera.zoom = 1; camera.panX = 0; camera.panY = 0; draw();
}

async function refreshMaps() {
  try {
    const result = await api('/api/maps');
    $('mapSelect').replaceChildren(...result.maps.map(m => {
      const option = document.createElement('option');
      option.value = m.name; option.textContent = `${m.name} · ${m.size_mb} MB`; return option;
    }));
    updateUi();
  } catch (error) { $('statusText').textContent = error.message; }
}

function setBadge(id, text, tone = '') {
  $(id).textContent = text;
  $(id).className = `badge ${tone}`;
}

function setNote(id, text, tone = '') {
  $(id).textContent = text;
  $(id).className = `note ${tone}`;
}

function updatePreparationUi(loco) {
  const selector = $('locoPreparePose');
  const options = loco.prepare_pose_options || [];
  const signature = JSON.stringify(options);
  if (selector.dataset.options !== signature) {
    selector.replaceChildren(...options.map(item => {
      const option = document.createElement('option');
      option.value = item.value;
      option.textContent = item.label;
      return option;
    }));
    selector.dataset.options = signature;
  }
  if (preparationPending) {
    const confirmed = loco.online && (preparationPending.action === 'prepare_pose'
      ? loco.prepare_pose === preparationPending.pose
      : ['PREPARING', 'HOLDING', 'POLICY'].includes(loco.stage));
    if (confirmed) {
      preparationPending = null;
    } else if (!loco.online || Date.now() - preparationPending.started > 4000
      || (preparationPending.action === 'prepare_pose' && !loco.prepare_pose_selectable)
      || ['DAMPED', 'FAULT'].includes(loco.stage)) {
      preparationError = '操作未确认，请查看当前姿态和运控状态后重试。';
      preparationPending = null;
    }
  }
  selector.value = preparationPending?.pose || loco.prepare_pose || '';
  selector.disabled = !loco.online || !loco.actions?.prepare_pose || !!preparationPending;
  if (preparationPending) {
    ['locoAcquire', 'locoPrepare', 'locoPolicy'].forEach(id => { $(id).disabled = true; });
  }
  let message = !loco.online ? '等待运控连接'
    : loco.prepare_pose_selectable ? '选择本身不动作；点击②准备姿态后执行并锁定。'
    : ['PREPARING', 'HOLDING', 'POLICY'].includes(loco.stage)
      ? '姿态已锁定，后续策略沿用此上半身姿态。' : '当前状态不可选择准备姿态。';
  if (preparationPending) message = preparationPending.action === 'prepare_pose'
    ? '等待运控确认所选姿态…' : '准备请求已发送，等待运控确认…';
  setNote('locoPreparePoseStatus', preparationError || message, preparationError ? 'bad' : '');
}

function updateModelSetUi(loco) {
  const selector = $('locoModelSet');
  const options = loco.model_set_options || [];
  const signature = JSON.stringify(options);
  if (selector.dataset.options !== signature) {
    selector.replaceChildren(...options.map(item => {
      const option = document.createElement('option');
      option.value = item.value; option.textContent = item.label;
      return option;
    }));
    selector.dataset.options = signature;
  }
  if (modelSetPending) {
    if (loco.online && loco.model_set === modelSetPending.value) {
      modelSetPending = null;
    } else if (!loco.online || !loco.model_set_selectable || Date.now() - modelSetPending.started > 4000) {
      modelSetError = '切换未确认，请检查当前模型组合后重试。';
      modelSetPending = null;
    }
  }
  selector.value = modelSetPending?.value || loco.model_set || '';
  selector.disabled = !loco.actions?.model_set || !!modelSetPending || !!preparationPending || !!lifecyclePending;
  if (modelSetPending) {
    ['locoAcquire', 'locoPrepare', 'locoPolicy', 'locoPreparePose'].forEach(id => { $(id).disabled = true; });
  }
  const message = modelSetPending ? '等待运控确认模型组合…'
    : !loco.online ? '等待运控连接'
    : loco.model_set_selectable ? '已加载并预热；选择后再准备，行走与推车模型成套切换。'
    : '本次模型组合已锁定。停止并复位后可重新选择。';
  setNote('locoModelSetStatus', modelSetError || message, modelSetError ? 'bad' : '');
}

function updateLocomotionUi(loco) {
  const actions = loco.actions || {};
  const warnings = loco.online ? (loco.warnings || []) : [];
  $('locoWarnings').hidden = !warnings.length;
  setNote('locoWarnings', warnings.length ? `运控提示 · ${warnings.length} 条\n` + warnings.map(item =>
    `${item.message}${item.active === false ? '（近期提示，已恢复）' : ''}`).join('\n') : '',
    warnings.some(item => item.level === 'error' && item.active !== false) ? 'bad' : 'warn');
  [['locoAcquire', 'acquire'], ['locoPrepare', 'prepare'], ['locoPolicy', 'policy'],
    ['locoZero', 'zero'], ['locoStop', 'stop'], ['locoReset', 'reset']].forEach(([id, action]) => {
    $(id).disabled = !statusFresh() || !actions[action] || (!!lifecyclePending && !['zero', 'stop'].includes(action));
  });
  updatePreparationUi(loco);
  updateModelSetUi(loco);
  if (!statusFresh() || lifecyclePending) {
    $('locoPreparePose').disabled = true;
    $('locoModelSet').disabled = true;
  }
  if (!loco.online) {
    setBadge('locoBadge', '未连接', 'bad');
    setNote('locoStatus', '运控执行进程未连接。确认 run_navigation.sh 已启动 nav/locomotion，且 Viser 网页没有占用底层桥接。', 'bad');
    return;
  }
  const stopped = ['DAMPED', 'FAULT'].includes(loco.stage);
  setBadge('locoBadge', loco.ready ? '策略运行' : loco.stage_text,
    loco.ready ? 'ok' : stopped ? 'bad' : loco.stage === 'WAITING' ? '' : 'busy');
  const lines = [`${loco.stage_text}${loco.simulation ? '（离线合成反馈）' : ''}`];
  const modelLabel = (loco.model_set_options || []).find(item => item.value === loco.model_set)?.label;
  if (modelLabel) lines.push(`当前模型：${modelLabel}`);
  if (loco.selected_model && loco.stage === 'POLICY') {
    const v = loco.velocity || [0, 0, 0];
    lines.push(`模型 ${loco.selected_model} · 策略命令 ${v.map(n => Number(n).toFixed(2)).join(' / ')}`);
    lines.push(loco.decelerating ? '正在平滑归零，随后切回 Balance'
      : loco.nav_command_fresh ? '正在执行 Nav2 速度命令' : '无新导航命令（保持零速）');
  }
  if (loco.preparation_warning) lines.push(loco.preparation_warning);
  if (loco.reason) lines.push(loco.reason);
  setNote('locoStatus', lines.join('\n'), loco.ready ? 'ok' : stopped ? 'bad' : '');
}

function updateUi() {
  if (!status.pose && ['SELECT_MAP', 'PROCESSING_MAP', 'LOADING_MAP', 'STARTING_LOCALIZATION', 'WAITING_INITIAL_POSE'].includes(status.phase)) {
    robotPose = null;
    initialCandidate = null;
  }
  goalPose = status.goal || goalPose;
  const loco = status.locomotion || { online: false, actions: {} };
  $('statusText').textContent = status.error ? `${status.message}：${status.error}` : (status.message || '等待状态');
  const fresh = statusFresh();
  const mapOperationAllowed = fresh && !mapBusy && !!$('mapSelect').value && !status.processing && !status.navigation_active;
  const requiresPolicy = status.initial_pose_requires_policy !== false;
  const bodyPolicyReady = !!loco.online && !!loco.healthy && loco.stage === 'POLICY';
  const initialPolicyReady = !requiresPolicy || (bodyPolicyReady && !loco.decelerating);
  const initialAllowed = fresh && !poseBusy && !!status.map_ready && !status.processing && !status.navigation_active && initialPolicyReady;
  const speed = status.navigation_speed || {};
  const goalAllowed = fresh && !poseBusy && !!status.localized && !!status.motion_enabled && !!loco.ready && !status.navigation_active && !loco.decelerating && speed.ready && !speed.pending;
  $('processMap').disabled = !mapOperationAllowed;
  $('processMap').textContent = status.map_ready ? '切换 / 重新处理所选 PCD' : '处理并装载所选 PCD';
  $('initialMode').disabled = !initialAllowed;
  $('initialPoseHint').textContent = !requiresPolicy ? '只读定位：确认机器人姿态稳定后，在地图上设置初始位置和朝向。'
    : !bodyPolicyReady ? '先完成准备姿态、启动混合策略，确认机器人落地站稳后再设置初始位姿。'
    : loco.decelerating ? '等待机器人停止并站稳后再设置初始位姿。'
    : '混合策略已启动。确认机器人已落地站稳，再设置初始位姿；运动门保持关闭直到定位完成。';
  $('goalMode').disabled = !goalAllowed;
  $('toolbarInitial').disabled = !initialAllowed;
  $('toolbarGoal').disabled = !goalAllowed;
  const gate = status.motion_gate || {};
  if (gate.suspended) $('statusText').textContent = '导航暂缓，正在归零并等待观测恢复';
  if (!status.error && status.phase === 'READY' && status.localized && !status.motion_enabled
      && (status.message || '').startsWith('AMCL 定位稳定')) {
    $('statusText').textContent = gate.can_enable ? 'AMCL 定位稳定；可启用运动门'
      : `AMCL 定位稳定；运动门等待：${(gate.blockers || []).join('；') || '状态确认'}`;
  }
  $('motionButton').disabled = motionBusy || (fresh && status.motion_enabled ? false : !fresh || !gate.can_enable);
  $('motionButton').textContent = status.motion_enabled ? '关闭 G1 运动安全门' : '启用 G1 运动安全门';
  $('motionButton').classList.toggle('enabled', !!status.motion_enabled);
  const blockers = fresh ? (gate.blockers || []) : ['导航状态已过期，等待连接恢复'];
  if (!fresh) $('statusText').textContent = '导航状态已过期，等待连接恢复；停止按钮仍可用';
  $('gateBlockers').hidden = (fresh && !!status.motion_enabled && !gate.suspended) || !blockers.length;
  $('gateBlockers').textContent = blockers.length ? `${gate.suspended ? '导航暂缓' : '运动门等待'}：\n· ${blockers.join('\n· ')}` : '';
  // The stop endpoint is idempotent; keep the safety control always available.
  $('stopButton').disabled = false;
  if ((mode === 'initial' && !initialAllowed) || (mode === 'goal' && !goalAllowed)) {
    dragStart = dragCurrent = null;
    setMode('view');
  }

  $('stepMap').className = status.map_ready ? 'done' : (status.processing ? 'active' : '');
  $('stepLoc').className = status.localized ? 'done' : (status.map_ready && initialPolicyReady ? 'active' : '');
  $('stepLoco').className = bodyPolicyReady ? 'done' : (status.map_ready ? 'active' : '');
  $('stepMotion').className = status.motion_enabled ? 'done' : (status.localized && loco.ready ? 'active' : '');
  $('stepNav').className = status.navigation_active ? 'active'
    : ((status.navigation_task || {}).state === 'SUCCEEDED' ? 'done' : '');

  setBadge('mapBadge', status.map_ready ? (status.selected_map || '已装载') : status.processing ? '处理中' : '未装载',
    status.map_ready ? 'ok' : status.processing ? 'busy' : '');
  setBadge('locBadge', status.localization_fault ? '定位故障' : status.localized ? 'READY'
    : status.phase === 'LOCALIZING' ? '修正中' : !status.map_ready ? '等待地图'
    : !initialPolicyReady ? '等待策略启动并站稳' : '等待初始位姿',
    status.localization_fault ? 'bad' : status.localized ? 'ok' : status.phase === 'LOCALIZING' ? 'busy' : '');
  setBadge('navBadge', status.navigation_active ? '导航中' : status.motion_enabled ? '可下发目标' : '运动门关闭',
    status.navigation_active ? 'busy' : status.motion_enabled ? 'ok' : '');
  updateLocomotionUi(loco);
  updateCartUi(loco);
  updateSpeedUi();

  const task = status.navigation_task || {};
  const navLines = [];
  if (status.navigation_active && Number.isFinite(Number(status.navigation_distance))) {
    navLines.push(`剩余距离 ${Number(status.navigation_distance).toFixed(2)} m`);
  }
  if (task.state && task.state !== 'IDLE' && !status.navigation_active) {
    navLines.push(`上次任务：${{ SUCCEEDED: '完成', CANCELED: '已取消', FAILED: '失败' }[task.state] || task.state}`);
  }
  if (status.last_safety_event) navLines.push(`最近安全事件：${status.last_safety_event}`);
  $('navStats').textContent = navLines.join('\n');

  if (status.metadata && status.metadata.xyz_min) {
    const m = status.metadata;
    $('mapStats').textContent = `XYZ 下限 [${m.xyz_min.map(v => v.toFixed(2)).join(', ')}]\nXYZ 上限 [${m.xyz_max.map(v => v.toFixed(2)).join(', ')}]\n地面 Z@原点 ${m.ground_z_at_origin.toFixed(3)} m · 占据高度 ${m.obstacle_height[0]} ~ ${m.obstacle_height[1]} m`;
  }
  const quality = status.localization_quality;
  const match = quality && quality.map_match;
  const correction = quality && Number.isFinite(Number(quality.correction_distance))
    ? ` · 修正 ${Number(quality.correction_distance).toFixed(2)} m/${(Number(quality.correction_yaw) * 180 / Math.PI).toFixed(1)}°`
    : '';
  $('poseReadout').textContent = robotPose
    ? `G1 Pose: x ${robotPose.x.toFixed(2)} · y ${robotPose.y.toFixed(2)} · yaw ${(robotPose.yaw * 180 / Math.PI).toFixed(1)}°${quality ? ` · AMCL ${status.localization_updates || 0}次/${status.localization_refinement_attempts || 0}轮 · cov ${Number(quality.covariance_xy).toFixed(3)} · 命中 ${match ? (100 * Number(match.within_20cm_ratio)).toFixed(1) : '--'}%${correction}` : ''}`
    : 'G1 Pose: --';
  const localizationScan = status.localization_scan || {};
  $('scanReadout').textContent = liveScan.receivedAt
    ? `/scan 避障: ${liveScan.count} hits/4 m · AMCL: ${localizationScan.points || 0} hits/8 m · 匹配 ${localizationScan.map_match ? (100 * Number(localizationScan.map_match.within_20cm_ratio)).toFixed(1) : '--'}% · ${((performance.now() - liveScan.receivedAt) / 1000).toFixed(1)} s`
    : '/scan: waiting';
}

function drawCart(dpr) {
  const pose = status.cart_pose;
  const reference = status.navigation_reference || {};
  if (!statusFresh() || !pose || reference.mode !== 'cart') return;
  const project = ([x, y]) => worldToScreen(
    pose.x + Math.cos(pose.yaw) * x - Math.sin(pose.yaw) * y,
    pose.y + Math.sin(pose.yaw) * x + Math.cos(pose.yaw) * y);
  const polygon = (vertices, fill = false) => {
    const points = (vertices || []).map(project);
    if (points.length < 3 || !points.every(Boolean)) return;
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    if (fill) ctx.fill();
    ctx.stroke();
  };
  ctx.save();
  if (visibility.footprint) {
    ctx.strokeStyle = '#ffb347'; ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    polygon(reference.footprint);
    ctx.setLineDash([]);
  }
  if (visibility.cart) {
    const model = reference.cart_model || {};
    const color = reference.ready ? '#63e6be' : '#a6b8b2';
    ctx.strokeStyle = color; ctx.fillStyle = 'rgba(99, 230, 190, 0.12)';
    ctx.lineWidth = 2 * dpr;
    polygon(model.body, true);
    for (const wheel of model.wheels || []) {
      const point = project(wheel.center);
      if (!point) continue;
      ctx.beginPath(); ctx.arc(...point, wheel.radius * viewTransform().scale, 0, Math.PI * 2);
      ctx.fillStyle = '#183e36'; ctx.fill(); ctx.stroke();
    }
    const handle = (model.handle || []).map(project);
    if (handle.length === 2 && handle.every(Boolean)) {
      ctx.lineWidth = 4 * dpr; ctx.beginPath(); ctx.moveTo(...handle[0]);
      ctx.lineTo(...handle[1]); ctx.stroke();
    }
    const degrees = Math.atan2(Math.sin(pose.yaw), Math.cos(pose.yaw)) * 180 / Math.PI;
    drawArrow(pose, color, 22 * dpr, `小车地图航向 ${degrees.toFixed(1)}°（位置估计）`);
  }
  ctx.restore();
}

const handLabels = ['拇指旋转', '拇指近端', '拇指远端', '中指近端', '中指远端', '食指近端', '食指远端'];
for (let joint = 0; joint < 14; joint++) {
  const label = document.createElement('label'); label.className = 'field';
  label.textContent = `${joint < 7 ? '左' : '右'}手 ${handLabels[joint % 7]}（rad）`;
  const input = document.createElement('input');
  input.id = `cartJoint${joint}`; input.type = 'number'; input.step = 'any'; input.disabled = true;
  input.onchange = () => {
    if (input.reportValidity() && Number.isFinite(input.valueAsNumber))
      cartRequest('hand_edit', { joint, angle_rad: input.valueAsNumber });
  };
  label.appendChild(input); $('cartHandEditor').appendChild(label);
}
function updateCartUi(loco) {
  const high = loco.high_policy || {}, hands = loco.hands || {}, cart = loco.cart || {};
  const reference = status.navigation_reference || {}, camera = high.camera || {};
  const gantry = reference.gantry_filter || {};
  $('gantryStatus').hidden = !gantry.enabled;
  if (gantry.enabled) {
    const regions = (gantry.boxes || []).map(box =>
      `身后 ${(-box.max_m[0]).toFixed(2)}–${(-box.min_m[0]).toFixed(2)} m，横向 ${box.min_m[1].toFixed(2)}–${box.max_m[1].toFixed(2)} m`).join('；');
    setNote('gantryStatus', `后方区域清空已开启（随机器人移动）\n${regions}\n区域内人员和物体不参与实时避障，区域外继续避障。${gantry.ready ? '' : '\n' + (gantry.reason || reference.reason || '等待避障扫描')}`, 'warn');
  }
  const actions = loco.actions || {};
  const reply = loco.cart_request || {};
  if (cartPending && reply.sequence === cartPending.id && reply.state !== 'queued') {
    cartError = reply.state === 'rejected' ? reply.error || '操作未执行' : '';
    if (reply.state === 'handled' && cartPending.action === 'cart_pose_source') cartPoseDirty = false;
    cartPending = null;
  } else if (cartPending && Date.now() - cartPending.started > 5000) {
    cartError = '操作未确认，请核对当前抓握及策略状态后重试';
    cartPending = null;
  }
  const busy = cartBusy || !!cartPending || !statusFresh();
  const poseSelect = $('cartPoseSource');
  if (!cartPoseDirty) poseSelect.value = cart.pose_source || 'grasp';
  poseSelect.disabled = busy || !!status.navigation_active || !actions.cart_pose_source;
  $('applyCartPoseSource').disabled = poseSelect.disabled || poseSelect.value === (cart.pose_source || 'grasp');
  setBadge('cartBadge', cart.attached ? '小车中心导航' : '普通导航', cart.attached ? 'ok' : '');
  document.querySelectorAll('[data-cart-action]').forEach(button => {
    button.disabled = busy || !!status.navigation_active || !actions[button.dataset.cartAction];
  });
  $('cartCompensation').disabled = busy || !!status.navigation_active || !actions.hand_compensation;
  $('cartCompensation').textContent = hands.compensation_enabled ? '关闭抓握补偿' : '启用抓握补偿';
  for (let joint = 0; joint < 14; joint++) {
    const input = $(`cartJoint${joint}`), side = joint < 7 ? 0 : 1, index = joint % 7;
    input.disabled = busy || !!status.navigation_active || !actions.hand_edit;
    if (hands.lower?.[side]) input.min = hands.lower[side][index];
    if (hands.upper?.[side]) input.max = hands.upper[side][index];
    if ((document.activeElement !== input || input.dataset.revision !== String(hands.grasp_revision)) && hands.grasp?.[side]) {
      input.value = String(hands.grasp[side][index]);
      input.dataset.revision = String(hands.grasp_revision);
    }
  }
  const handButton = document.querySelector('[data-cart-action="hand_toggle"]');
  handButton.textContent = !hands.active ? (hands.ever_started ? '重新使能 Dex3 并归零' : '使能 Dex3 并抓握')
    : ['grasp', 'grasp_compensated'].includes(hands.pose) ? '松手 / 手指归零' : '抓握小车';
  const lines = [high.enabled ? '推车高层运行中' : '推车高层未启用',
    `Dex3：${hands.pose || '未连接'}${hands.moving ? '（动作中）' : ''}`];
  lines.push(cart.pose_source === 'apriltag' ? '位姿方式：视觉 AprilTag · URDF 相机外参' : '位姿方式：固定抓握（无滑移假设）');
  if (high.history_required) lines.push(`策略观测 ${high.history_frames || 0}/${high.history_required}`);
  lines.push(camera.fresh && Number.isFinite(camera.yaw_rad)
    ? `小车相对机身航向 ${(camera.yaw_rad * 180 / Math.PI).toFixed(1)}°（实测）` : '等待新鲜的小车航向');
  if (cart.valid && cart.position) lines.push(`小车中心相对机身：${cart.position.slice(0, 2).map(v => v.toFixed(3)).join(', ')} m（${cart.position_source || '抓握几何估计'}）`);
  if (!cart.valid && cart.reason) lines.push(`位姿未就绪：${cart.reason}`);
  if (Number.isFinite(cart.rmse_px)) lines.push(`视觉重投影误差：${cart.rmse_px.toFixed(3)} px`);
  if (reference.mode === 'cart') lines.push(`车体过滤：${reference.removed_points || 0} 点；${reference.ready ? '导航参考已就绪' : reference.reason || '等待参考点'}`);
  const calibration = high.zero_calibration || {};
  const zeroLines = [calibration.message || high.zero_reason || '等待航向标定'];
  if (calibration.status === 'collecting') zeroLines.push(`正在采集 ${calibration.samples || 0}/${calibration.required || 30} 帧`);
  if (Number.isFinite(camera.yaw_offset_rad)) zeroLines.push(`当前零位偏置 ${(camera.yaw_offset_rad * 180 / Math.PI).toFixed(2)}°`);
  setNote('cartZeroStatus', zeroLines.join('\n'), calibration.status === 'collecting' ? 'warn' : calibration.status === 'done' ? 'ok' : '');
  setNote('cartHandStatus', hands.editor_message || (hands.grasp_dirty ? '当前抓握修改尚未保存' : hands.blocking_reason || '抓握参数沿用当前 Viser 配置'), hands.file_busy || hands.grasp_dirty ? 'warn' : '');
  if (cartPending) lines.push('操作已发送，等待运控确认…');
  if (loco.upper_pd_transitioning) lines.push('上肢增益正在切换，完成后可启动导航');
  const issue = cartError || cart.error || loco.pushcart_error || high.camera_process?.error || high.warmup_error || high.reason || high.enable_reason;
  if (issue) lines.push(issue);
  setNote('cartStatus', lines.join('\n'), issue ? 'warn' : high.enabled ? 'ok' : '');
  $('goalMode').textContent = cart.attached ? '设置小车中心导航目标' : '设置导航目标';
}
async function cartRequest(action, values = {}) {
  if (cartBusy || cartPending || !statusFresh()) return;
  cartBusy = true; cartError = ''; updateCartUi(status.locomotion || {});
  try {
    const result = await api(`/api/locomotion?${new URLSearchParams({ action, ...values })}`);
    cartPending = { id: result.request_id, action, started: Date.now() };
    await refreshStatus();
  } catch (error) { cartError = error.message; }
  finally { cartBusy = false; updateUi(); draw(); }
}
document.querySelectorAll('[data-cart-action]').forEach(button => {
  button.onclick = () => cartRequest(button.dataset.cartAction);
});
$('cartCompensation').onclick = () => cartRequest('hand_compensation', {
  enabled: !status.locomotion?.hands?.compensation_enabled,
});
$('cartPoseSource').onchange = () => { cartPoseDirty = true; updateCartUi(status.locomotion || {}); };
$('applyCartPoseSource').onclick = () => cartRequest('cart_pose_source', { source: $('cartPoseSource').value });

function updateSpeedUi() {
  const speed = status.navigation_speed || {};
  const busy = speedRequestBusy || speed.pending || !statusFresh();
  if (!speedDirty && !speedRequestBusy && speed.mode) {
    const shown = speed.requested || speed;
    $('speedMode').value = shown.mode;
    $('speedValue').value = Number(shown.speed).toFixed(2);
    $('speedSlider').value = shown.speed;
  }
  $('speedMode').disabled = !!busy;
  $('speedValue').disabled = !!busy;
  $('speedSlider').disabled = !!busy;
  $('applySpeed').disabled = !!busy || !speed.mode;
  const fixed = $('speedMode').value === 'fixed';
  const approach = status.locomotion?.cart?.attached ? '0.8' : '0.6';
  $('speedLabel').textContent = fixed ? '巡航速度（m/s）' : '速度上限（m/s）';
  $('speedHint').textContent = fixed
    ? `普通路段优先保持设定速度；末段 ${approach} m 内减速，到达 15 cm / 10° 范围后归零。转弯或遇障碍时允许减速、停车。`
    : `由 Nav2 根据路径和障碍自动调速，不超过设定上限；末段 ${approach} m 内进一步减速，到达 15 cm / 10° 范围后归零。`;
  const lines = [];
  if (speed.ready) lines.push(`已应用：${speed.mode === 'fixed' ? '固定巡航' : '自动调速上限'} ${Number(speed.speed).toFixed(2)} m/s`);
  if (busy) lines.push('正在等待 Nav2 确认…');
  else if (speedDirty) lines.push('当前修改尚未应用');
  else if (!speed.ready) lines.push('等待 Nav2 速度控制就绪…');
  const error = speedRequestError || speed.error;
  if (error) lines.push(error);
  if (speed.save_warning) lines.push(speed.save_warning);
  setNote('speedStatus', lines.join('\n'), error ? 'bad' : busy || speedDirty || speed.save_warning ? 'warn' : speed.ready ? 'ok' : '');
}

function editSpeed(source) {
  speedDirty = true;
  speedRequestError = '';
  if (source === 'slider') $('speedValue').value = Number($('speedSlider').value).toFixed(2);
  if (source === 'value' && $('speedValue').validity.valid) $('speedSlider').value = $('speedValue').value;
  updateSpeedUi();
}
$('speedMode').onchange = () => editSpeed('mode');
$('speedValue').oninput = () => editSpeed('value');
$('speedSlider').oninput = () => editSpeed('slider');
$('applySpeed').onclick = async () => {
  if (speedRequestBusy || !statusFresh()) return;
  if (!$('speedValue').reportValidity() || !Number.isFinite($('speedValue').valueAsNumber)) return;
  const params = new URLSearchParams({ mode: $('speedMode').value, speed: $('speedValue').value });
  speedRequestBusy = true;
  speedRequestError = '';
  updateSpeedUi();
  try {
    const result = await api(`/api/navigation/speed?${params}`);
    status.navigation_speed = result.navigation_speed;
    speedDirty = false;
  } catch (error) { speedRequestError = error.message; }
  finally { speedRequestBusy = false; updateSpeedUi(); }
};

$('processMap').onclick = async () => {
  if (mapBusy || !statusFresh() || !$('mapSelect').value) return;
  if (!['resolution', 'minHeight', 'maxHeight', 'obstacleRadius', 'freeRadius'].every(id => $(id).reportValidity())) return;
  if (Number($('maxHeight').value) <= Number($('minHeight').value)) { alert('障碍物高度上限必须大于下限'); return; }
  const replacing = !!status.map_ready;
  if (replacing && !window.confirm('切换或重新处理 PCD 会清除当前 AMCL 定位、路径和运动门状态，确定继续吗？')) return;
  const params = new URLSearchParams({
    name: $('mapSelect').value, resolution: $('resolution').value, ground_z: $('groundZ').value,
    obstacle_min_height: $('minHeight').value, obstacle_max_height: $('maxHeight').value,
    obstacle_radius: $('obstacleRadius').value, free_radius: $('freeRadius').value,
  });
  if (replacing) params.set('replace', 'true');
  mapBusy = true; updateUi();
  try {
    await api(`/api/map/process?${params}`);
    robotPose = null; initialCandidate = null; goalPose = null;
    paths = { global: [], local: [] };
    resetCamera();
    await refreshStatus();
  } catch (error) { alert(error.message); }
  finally { mapBusy = false; updateUi(); }
};

const locomotionConfirm = {
  acquire: '接管底层会释放原厂控制器，机器人此时没有姿态支撑。请确认已吊装或有人扶稳，确定接管吗？',
  prepare: '机器人将平滑移到准备姿态（至少 3 秒）。确认周围安全吗？',
  policy: '将启动 Balance / Homie 混合策略，速度命令从零开始。确认可以放开吊装/扶持吗？',
  stop: '阻尼停止会立即卸掉支撑力，机器人可能倒下。确认已吊装或扶稳吗？',
};
async function locomotionRequest(action) {
  if (!statusFresh() || (action === 'prepare' && preparationPending)) return;
  if ((lifecyclePending || modelSetPending) && !['zero', 'stop'].includes(action)) return;
  const pose = status.locomotion?.prepare_pose;
  const label = status.locomotion?.prepare_pose_options?.find(item => item.value === pose)?.label;
  const confirmation = action === 'prepare' && label
    ? `已选：${label}。\n${locomotionConfirm[action]}` : locomotionConfirm[action];
  if (confirmation && !window.confirm(confirmation)) return;
  if (action === 'prepare') {
    preparationError = '';
    preparationPending = { action, started: Date.now() };
    updateLocomotionUi(status.locomotion || {});
  }
  const pending = { action };
  lifecyclePending = pending; updateUi();
  try {
    await api(`/api/locomotion?action=${action}`);
    await refreshStatus();
  } catch (error) {
    if (action === 'prepare') {
      preparationPending = null;
      preparationError = error.message;
      updateLocomotionUi(status.locomotion || {});
    }
    alert(error.message);
  } finally {
    if (lifecyclePending === pending) lifecyclePending = null;
    updateUi();
  }
}
$('locoPreparePose').onchange = async () => {
  if (preparationPending || modelSetPending || !status.locomotion?.actions?.prepare_pose) return;
  const pose = $('locoPreparePose').value;
  preparationError = '';
  preparationPending = { action: 'prepare_pose', pose, started: Date.now() };
  updateLocomotionUi(status.locomotion || {});
  try {
    await api(`/api/locomotion?action=prepare_pose&pose=${encodeURIComponent(pose)}`);
    await refreshStatus();
  } catch (error) {
    preparationPending = null;
    preparationError = error.message;
  }
  updateUi();
};
$('locoModelSet').onchange = async () => {
  if (!statusFresh() || modelSetPending || preparationPending || lifecyclePending
      || !status.locomotion?.actions?.model_set) return;
  const value = $('locoModelSet').value;
  modelSetError = '';
  modelSetPending = { value, started: Date.now() };
  updateUi();
  try {
    await api(`/api/locomotion?action=model_set&model_set=${encodeURIComponent(value)}`);
    await refreshStatus();
  } catch (error) {
    modelSetPending = null;
    modelSetError = error.message;
  }
  updateUi();
};
[['locoAcquire', 'acquire'], ['locoPrepare', 'prepare'], ['locoPolicy', 'policy'],
  ['locoZero', 'zero'], ['locoStop', 'stop'], ['locoReset', 'reset']].forEach(([id, action]) => {
  $(id).onclick = () => locomotionRequest(action);
});

$('initialMode').onclick = () => setMode('initial');
$('goalMode').onclick = () => setMode('goal');
$('panMode').onclick = () => setMode('view');
$('toolbarInitial').onclick = () => setMode('initial');
$('toolbarGoal').onclick = () => setMode('goal');
$('focusRobot').onclick = resetCamera;
$('sidebarToggle').onclick = () => {
  $('layout').classList.toggle('collapsed');
  // Wait for the grid transition before resizing the canvas backing store.
  setTimeout(draw, 220);
};
document.querySelectorAll('[data-display]').forEach(input => {
  input.onchange = () => { visibility[input.dataset.display] = input.checked; draw(); };
});
$('motionButton').onclick = async () => {
  if (motionBusy || !statusFresh()) return;
  const enabled = !status.motion_enabled;
  motionBusy = true; updateUi();
  try { await api(`/api/motion?enabled=${enabled}`); await refreshStatus(); }
  catch (error) { alert(error.message); }
  finally { motionBusy = false; updateUi(); }
};
$('stopButton').onclick = async () => {
  try { await api('/api/stop'); await refreshStatus(); } catch (error) { alert(error.message); }
};
window.addEventListener('resize', draw);
setInterval(async () => {
  if (!statusPolling) {
    statusPolling = true;
    try { await refreshStatus(); } catch (_) {} finally { statusPolling = false; }
  }
}, 500);
setInterval(() => {
  if (!statusFresh()) updateUi();
  if (liveScan.receivedAt && performance.now() - liveScan.receivedAt > 1500) {
    $('scanStats').textContent = '实时雷达数据超时，请检查 /scan 与点云高度切片';
  }
  draw();
}, 500);
refreshMaps(); connectRosbridge(); setMode('view'); draw();
