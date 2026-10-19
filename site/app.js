const $ = (id) => document.getElementById(id);

let songs = null;
let songsTableBuilt = false;
let ctx = null;
let node = null;
let gain = null;
let analyser = null;
let compiled = false;
let playing = false;
let pendingPlay = false;
let rafId = 0;
let pos = 0;
let playPos0 = 0;
let playStart = 0;
let playRate = 22050;
let seeking = false;

async function loadSongs() {
  const res = await fetch('songs.json');
  if (!res.ok) throw new Error('songs.json HTTP ' + res.status);
  songs = await res.json();
}

function currentSong() {
  const id = $('song').value;
  return songs.songs.find((s) => s.id === id);
}

function fillTextarea(song) {
  $('code').value = '';
  $('status').textContent = 'loading ' + song.file + ' ...';
  fetch('songs/' + song.file)
    .then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    })
    .then((text) => {
      $('code').value = text;
      $('status').textContent = song.file + ' loaded, ' + text.length + ' bytes. Press Compile or Play.';
    })
    .catch((e) => {
      $('status').textContent = 'load failed: ' + e.message;
    });
}

async function ensureAudio() {
  if (ctx) return true;
  try {
    ctx = new AudioContext({ sampleRate: 48000 });
    await ctx.audioWorklet.addModule('processor.js');
    node = new AudioWorkletNode(ctx, 'bytebeat', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    gain = ctx.createGain();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    gain.gain.value = $('vol').value / 100;
    node.connect(gain);
    gain.connect(analyser);
    analyser.connect(ctx.destination);
    node.port.onmessage = (e) => {
      if (e.data.type === 'loaded') {
        if (e.data.err) {
          compiled = false;
          pendingPlay = false;
          $('status').textContent = 'compile error: ' + e.data.err;
        } else {
          compiled = true;
          $('status').textContent = 'compiled';
          if (pendingPlay) {
            pendingPlay = false;
            startPlayback();
          }
        }
      }
    };
    return true;
  } catch (e) {
    $('status').textContent = 'audio init failed: ' + e.message;
    return false;
  }
}

async function compile() {
  const ok = await ensureAudio();
  if (!ok) return;
  if (ctx.state === 'suspended') ctx.resume();
  compiled = false;
  node.port.postMessage({
    type: 'load',
    code: $('code').value,
    mode: parseInt($('mode').value, 10),
    rate: parseFloat($('rate').value),
  });
  $('status').textContent = 'compiling ...';
}

function startPlayback() {
  if (!compiled) {
    pendingPlay = true;
    compile();
    return;
  }
  if (ctx.state === 'suspended') ctx.resume();
  playRate = parseFloat($('rate').value) || 22050;
  playPos0 = pos;
  playStart = performance.now();
  node.port.postMessage({ type: 'play', t: Math.round(pos) });
  playing = true;
  $('status').textContent = 'playing';
  drawScope();
}

function stopPlayback() {
  const wasPlaying = playing;
  if (wasPlaying) pos = playPos0 + ((performance.now() - playStart) / 1000) * playRate;
  playing = false;
  pendingPlay = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  if (node) node.port.postMessage({ type: 'stop' });
  compiled = false;
  seeking = false;
  if (wasPlaying) {
    const c = $('scope');
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }
  syncSeek();
}

function drawScope() {
  if (!playing) return;
  rafId = requestAnimationFrame(drawScope);
  const c = $('scope');
  const g = c.getContext('2d');
  const w = c.width;
  const h = c.height;
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  g.fillStyle = '#0d1117';
  g.fillRect(0, 0, w, h);
  const mid = h / 2;
  g.strokeStyle = '#30363d';
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, mid);
  g.lineTo(w, mid);
  g.stroke();
  g.strokeStyle = '#7ee787';
  g.lineWidth = 1.5;
  g.shadowColor = 'rgba(126,231,135,0.55)';
  g.shadowBlur = 6;
  g.beginPath();
  for (let i = 0; i < buf.length; i++) {
    const x = (i / (buf.length - 1)) * w;
    const y = mid - buf[i] * (mid - 2);
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.stroke();
  g.lineTo(w, mid);
  g.lineTo(0, mid);
  g.closePath();
  g.fillStyle = 'rgba(126,231,135,0.12)';
  g.shadowBlur = 0;
  g.fill();
  if (playing && !seeking) pos = playPos0 + ((performance.now() - playStart) / 1000) * playRate;
  if (!seeking) syncSeek();
}

function fmtPos(p) {
  const t = Math.max(0, p) / (playRate || 22050);
  const tenths = Math.round(t * 10);
  const m = Math.floor(tenths / 600);
  const s = ((tenths % 600) / 10).toFixed(1);
  return m + ':' + (s < 10 ? '0' : '') + s;
}

function updatePos() {
  $('pos').textContent = fmtPos(pos);
}

function syncSeek() {
  const el = $('seek');
  if (el) el.value = String(Math.min(120, pos / (playRate || 22050)));
  updatePos();
}

function buildParams() {
  const params = Object.getOwnPropertyNames(Math);
  params.push('int', 'window');
  const values = Object.values(Math);
  values.push(Math.floor, globalThis);
  return [params, values];
}

function transformSample(mode, v) {
  if (mode === 0) return Number(v & 255) / 127.5 - 1;
  if (mode === 1) return (((Number(v & 255) + 128) & 255) - 128) / 128;
  const x = Number(v);
  if (!Number.isFinite(x)) return 0;
  return x > 1 ? 1 : x < -1 ? -1 : x;
}

function renderMain(code, mode, rate, seconds) {
  const n = Math.floor(rate * seconds);
  const out = new Float32Array(n);
  const [params, values] = buildParams();
  let fn;
  if (mode === 3) {
    fn = new Function(...params, code).bind(globalThis, ...values);
    fn = fn();
    fn(0, rate);
  } else {
    fn = new Function(...params, 't', 'return 0,\n' + code + ';').bind(globalThis, ...values);
    try {
      fn(0);
    } catch (e) {}
  }
  for (let i = 0; i < n; i++) {
    let v;
    try {
      v = mode === 3 ? fn(i / rate, rate) : fn(i);
    } catch (e) {
      v = 0;
    }
    out[i] = transformSample(mode, v);
  }
  return out;
}

function encodeWav(data, rate) {
  const n = data.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const w = (off, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
  };
  w(0, 'RIFF');
  dv.setUint32(4, 36 + n * 2, true);
  w(8, 'WAVE');
  w(12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  w(36, 'data');
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, data[i])) * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

async function exportWav() {
  const song = currentSong();
  const rate = parseFloat($('rate').value);
  const mode = parseInt($('mode').value, 10);
  $('status').textContent = 'rendering 15s ...';
  await new Promise((r) => setTimeout(r, 30));
  try {
    const data = renderMain($('code').value, mode, rate, 15);
    const blob = encodeWav(data, rate);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (song ? song.id : 'export') + '.wav';
    a.click();
    URL.revokeObjectURL(a.href);
    $('status').textContent = 'wav exported';
  } catch (e) {
    $('status').textContent = 'render error: ' + e.message;
  }
}

async function buildDollchanLink() {
  const base = 'https://dollchan.net/bytebeat/#4';
  try {
    const head = new Uint8Array(5);
    head[0] = parseInt($('mode').value, 10);
    new DataView(head.buffer).setFloat32(1, parseFloat($('rate').value), true);
    const stream = new Blob([$('code').value]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    const comp = new Uint8Array(await new Response(stream).arrayBuffer());
    const all = new Uint8Array(head.length + comp.length);
    all.set(head);
    all.set(comp, head.length);
    let bin = '';
    for (let i = 0; i < all.length; i += 0x8000) {
      bin += String.fromCharCode.apply(null, all.subarray(i, i + 0x8000));
    }
    $('dollchanLink').href = base + btoa(bin).replace(/=/g, '');
    $('status').textContent = 'link built, click the Dollchan link to open it';
    return true;
  } catch (e) {
    $('dollchanLink').href = 'https://dollchan.net/bytebeat/';
    $('status').textContent = 'link build failed: ' + e.message;
    return false;
  }
}

function buildSongsTable() {
  if (songsTableBuilt) return;
  songsTableBuilt = true;
  const tb = $('songsTable').querySelector('tbody');
  for (const s of songs.songs) {
    const tr = document.createElement('tr');
    for (const c of [s.id, s.modeName, s.rate + ' Hz', s.bytes.toLocaleString(), s.corr.toFixed(6)]) {
      const td = document.createElement('td');
      td.textContent = c;
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const btn = document.createElement('button');
    btn.textContent = 'Load';
    btn.onclick = () => {
      $('song').value = s.id;
      selectSong();
    };
    td.appendChild(btn);
    tr.appendChild(td);
    tb.appendChild(tr);
  }
}

function selectSong() {
  const s = currentSong();
  if (!s) return;
  $('mode').value = String(s.mode);
  $('rate').value = String(s.rate);
  compiled = false;
  playing = false;
  seeking = false;
  pos = 0;
  playPos0 = 0;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  if (node) node.port.postMessage({ type: 'stop' });
  syncSeek();
  fillTextarea(s);
}

function init() {
  loadSongs()
    .then(() => {
      const sel = $('song');
      sel.innerHTML = '';
      for (const s of songs.songs) {
        const o = document.createElement('option');
        o.value = s.id;
        o.textContent = s.id + ' (' + s.modeName + ')';
        sel.appendChild(o);
      }
      sel.value = songs.default || songs.songs[0].id;
      const s = currentSong();
      $('mode').value = String(s.mode);
      $('rate').value = String(s.rate);
      fillTextarea(s);
      buildSongsTable();
    })
    .catch((e) => {
      $('status').textContent = 'songs.json failed: ' + e.message;
    });
  $('song').onchange = selectSong;
  $('mode').onchange = () => {
    const wasPlaying = playing;
    if (wasPlaying) stopPlayback();
    compiled = false;
    if (wasPlaying) {
      pendingPlay = true;
      compile();
    } else {
      syncSeek();
    }
  };
  $('rate').onchange = () => {
    const wasPlaying = playing;
    if (wasPlaying) stopPlayback();
    compiled = false;
    playRate = parseFloat($('rate').value) || 22050;
    if (wasPlaying) {
      pendingPlay = true;
      compile();
    } else {
      syncSeek();
    }
  };
  $('compileBtn').onclick = compile;
  $('playBtn').onclick = () => {
    if (!playing) startPlayback();
  };
  $('stopBtn').onclick = stopPlayback;
  $('wavBtn').onclick = exportWav;
  $('linkBtn').onclick = buildDollchanLink;
  $('vol').oninput = () => {
    if (gain) gain.gain.value = $('vol').value / 100;
  };
  $('seek').oninput = () => {
    seeking = true;
    pos = parseFloat($('seek').value) * (playRate || 22050);
    updatePos();
  };
  $('seek').onchange = () => {
    seeking = false;
    playPos0 = pos;
    playStart = performance.now();
    if (playing && node) node.port.postMessage({ type: 'play', t: Math.round(pos) });
  };
  syncSeek();
}

init();
