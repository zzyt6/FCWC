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
  globalPath: true, localPath: true, robot: true, radar: true,
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

const $ = id => document.getElementById(id);
const api = async path => {
  const response = await fetch(path, { cache: 'no-store' });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || '请求失败');
  return data;
};

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
      status = JSON.parse(packet.msg.data);
      updateUi();
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
  if (!layers.map) return;
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
  try {
    if (mode === 'initial') {
      if (status.localized && !window.confirm('当前 AMCL 定位仍然有效。重新设置初始位姿会关闭运动门并重新定位，确定继续吗？')) {
        setMode('view');
        dragStart = dragCurrent = null;
        draw();
        return;
      }
      if (status.localized) params.set('replace', 'true');
      await api(`/api/initial_pose?${params}`);
      initialCandidate = targetPose;
    } else if (mode === 'goal') {
      paths = { global: [], local: [] };
      draw();
      await api(`/api/goal?${params}`);
      goalPose = targetPose;
    }
    setMode('view');
  } catch (error) { alert(error.message); }
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
    $('mapSelect').innerHTML = result.maps.map(m => `<option value="${m.name}">${m.name} · ${m.size_mb} MB</option>`).join('');
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

function updateLocomotionUi(loco) {
  const actions = loco.actions || {};
  const warnings = loco.online ? (loco.warnings || []) : [];
  $('locoWarnings').hidden = !warnings.length;
  setNote('locoWarnings', warnings.length ? `运控提示 · ${warnings.length} 条\n` + warnings.map(item =>
    `${item.message}${item.active === false ? '（近期提示，已恢复）' : ''}`).join('\n') : '',
    warnings.some(item => item.level === 'error' && item.active !== false) ? 'bad' : 'warn');
  [['locoAcquire', 'acquire'], ['locoPrepare', 'prepare'], ['locoPolicy', 'policy'],
    ['locoZero', 'zero'], ['locoStop', 'stop'], ['locoReset', 'reset']].forEach(([id, action]) => {
    $(id).disabled = !actions[action];
  });
  updatePreparationUi(loco);
  if (!loco.online) {
    setBadge('locoBadge', '未连接', 'bad');
    setNote('locoStatus', '运控执行进程未连接。确认 run_navigation.sh 已启动 nav/locomotion，且 Viser 网页没有占用底层桥接。', 'bad');
    return;
  }
  const stopped = ['DAMPED', 'FAULT'].includes(loco.stage);
  setBadge('locoBadge', loco.ready ? '策略运行' : loco.stage_text,
    loco.ready ? 'ok' : stopped ? 'bad' : loco.stage === 'WAITING' ? '' : 'busy');
  const lines = [`${loco.stage_text}${loco.simulation ? '（离线合成反馈）' : ''}`];
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
  const mapOperationAllowed = !status.processing && !status.navigation_active;
  const initialAllowed = !!status.map_ready && !status.processing && !status.navigation_active;
  const speed = status.navigation_speed || {};
  const goalAllowed = !!status.localized && !!status.motion_enabled && !status.navigation_active && !loco.decelerating && speed.ready && !speed.pending;
  $('processMap').disabled = !mapOperationAllowed;
  $('processMap').textContent = status.map_ready ? '切换 / 重新处理所选 PCD' : '处理并装载所选 PCD';
  $('initialMode').disabled = !initialAllowed;
  $('goalMode').disabled = !goalAllowed;
  $('toolbarInitial').disabled = !initialAllowed;
  $('toolbarGoal').disabled = !goalAllowed;
  const gate = status.motion_gate || {};
  $('motionButton').disabled = status.motion_enabled ? false : !gate.can_enable;
  $('motionButton').textContent = status.motion_enabled ? '关闭 G1 运动安全门' : '启用 G1 运动安全门';
  $('motionButton').classList.toggle('enabled', !!status.motion_enabled);
  const blockers = gate.blockers || [];
  $('gateBlockers').hidden = !!status.motion_enabled || !blockers.length;
  $('gateBlockers').textContent = blockers.length ? `运动门等待：\n· ${blockers.join('\n· ')}` : '';
  // The stop endpoint is idempotent; keep the safety control always available.
  $('stopButton').disabled = false;
  if ((mode === 'initial' && !initialAllowed) || (mode === 'goal' && !goalAllowed)) {
    dragStart = dragCurrent = null;
    setMode('view');
  }

  $('stepMap').className = status.map_ready ? 'done' : (status.processing ? 'active' : '');
  $('stepLoc').className = status.localized ? 'done' : (status.map_ready ? 'active' : '');
  $('stepLoco').className = loco.ready ? 'done' : (loco.online && !['DAMPED', 'FAULT'].includes(loco.stage) ? 'active' : '');
  $('stepMotion').className = status.motion_enabled ? 'done' : (status.localized && loco.ready ? 'active' : '');
  $('stepNav').className = status.navigation_active ? 'active'
    : ((status.navigation_task || {}).state === 'SUCCEEDED' ? 'done' : '');

  setBadge('mapBadge', status.map_ready ? (status.selected_map || '已装载') : status.processing ? '处理中' : '未装载',
    status.map_ready ? 'ok' : status.processing ? 'busy' : '');
  setBadge('locBadge', status.localization_fault ? '定位故障' : status.localized ? 'READY'
    : status.phase === 'LOCALIZING' ? '修正中' : status.map_ready ? '等待初始位姿' : '等待地图',
    status.localization_fault ? 'bad' : status.localized ? 'ok' : status.phase === 'LOCALIZING' ? 'busy' : '');
  setBadge('navBadge', status.navigation_active ? '导航中' : status.motion_enabled ? '可下发目标' : '运动门关闭',
    status.navigation_active ? 'busy' : status.motion_enabled ? 'ok' : '');
  updateLocomotionUi(loco);
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

function updateSpeedUi() {
  const speed = status.navigation_speed || {};
  const busy = speedRequestBusy || speed.pending;
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
  $('speedLabel').textContent = fixed ? '巡航速度（m/s）' : '速度上限（m/s）';
  $('speedHint').textContent = fixed
    ? '普通路段优先保持设定速度；末段 0.6 m 内减速，到达 15 cm / 10° 范围后归零。转弯或遇障碍时允许减速、停车。'
    : '由 Nav2 根据路径和障碍自动调速，不超过设定上限；末段 0.6 m 内进一步减速，到达 15 cm / 10° 范围后归零。';
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
  const replacing = !!status.map_ready;
  if (replacing && !window.confirm('切换或重新处理 PCD 会清除当前 AMCL 定位、路径和运动门状态，确定继续吗？')) return;
  const params = new URLSearchParams({
    name: $('mapSelect').value, resolution: $('resolution').value, ground_z: $('groundZ').value,
    obstacle_min_height: $('minHeight').value, obstacle_max_height: $('maxHeight').value,
    obstacle_radius: $('obstacleRadius').value, free_radius: $('freeRadius').value,
  });
  if (replacing) params.set('replace', 'true');
  try {
    robotPose = null; initialCandidate = null; goalPose = null;
    paths = { global: [], local: [] };
    resetCamera();
    await api(`/api/map/process?${params}`);
  } catch (error) { alert(error.message); }
};

const locomotionConfirm = {
  acquire: '接管底层会释放原厂控制器，机器人此时没有姿态支撑。请确认已吊装或有人扶稳，确定接管吗？',
  prepare: '机器人将平滑移到准备姿态（至少 3 秒）。确认周围安全吗？',
  policy: '将启动 Balance / Homie 混合策略，速度命令从零开始。确认可以放开吊装/扶持吗？',
  stop: '阻尼停止会立即卸掉支撑力，机器人可能倒下。确认已吊装或扶稳吗？',
};
async function locomotionRequest(action) {
  if (action === 'prepare' && preparationPending) return;
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
  try {
    await api(`/api/locomotion?action=${action}`);
    status = (await api('/api/status')).status;
    updateUi();
  } catch (error) {
    if (action === 'prepare') {
      preparationPending = null;
      preparationError = error.message;
      updateLocomotionUi(status.locomotion || {});
    }
    alert(error.message);
  }
}
$('locoPreparePose').onchange = async () => {
  if (preparationPending || !status.locomotion?.actions?.prepare_pose) return;
  const pose = $('locoPreparePose').value;
  preparationError = '';
  preparationPending = { action: 'prepare_pose', pose, started: Date.now() };
  updateLocomotionUi(status.locomotion || {});
  try {
    await api(`/api/locomotion?action=prepare_pose&pose=${encodeURIComponent(pose)}`);
    status = (await api('/api/status')).status;
  } catch (error) {
    preparationPending = null;
    preparationError = error.message;
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
  try { await api(`/api/motion?enabled=${status.motion_enabled ? 'false' : 'true'}`); } catch (error) { alert(error.message); }
};
$('stopButton').onclick = async () => { try { await api('/api/stop'); } catch (error) { alert(error.message); } };
window.addEventListener('resize', draw);
setInterval(async () => {
  try { status = (await api('/api/status')).status; updateUi(); } catch (_) {}
  if (liveScan.receivedAt && performance.now() - liveScan.receivedAt > 1500) {
    $('scanStats').textContent = '实时雷达数据超时，请检查 /scan 与点云高度切片';
  }
  draw();
}, 1000);
refreshMaps(); connectRosbridge(); setMode('view'); draw();
