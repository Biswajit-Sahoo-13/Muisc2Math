const buildParams = () => {
  const params = Object.getOwnPropertyNames(Math);
  params.push('int', 'window');
  const values = Object.values(Math);
  values.push(Math.floor, globalThis);
  return [params, values];
};

function transform(mode, v) {
  if (mode === 0) return Number(v & 255) / 127.5 - 1;
  if (mode === 1) return (((Number(v & 255) + 128) & 255) - 128) / 128;
  const x = Number(v);
  if (!Number.isFinite(x)) return 0;
  return x > 1 ? 1 : x < -1 ? -1 : x;
}

class ByteBeatProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.fn = null;
    this.t = 0;
    this.acc = 0;
    this.songRate = 22050;
    this.mode = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'load') {
        this.fn = null;
        this.t = 0;
        this.acc = 0;
        this.songRate = m.rate;
        this.mode = m.mode;
        let err = null;
        try {
          const [params, values] = buildParams();
          if (this.mode === 3) {
            let f = new Function(...params, m.code).bind(globalThis, ...values);
            f = f();
            f(0, this.songRate);
            this.fn = f;
          } else {
            const f = new Function(...params, 't', 'return 0,\n' + m.code + ';').bind(globalThis, ...values);
            try {
              f(0);
            } catch (e2) {
              err = String(e2);
            }
            this.fn = f;
          }
        } catch (e2) {
          err = String(e2);
        }
        this.port.postMessage({ type: 'loaded', err });
      } else if (m.type === 'play') {
        this.t = m.t | 0;
        this.acc = 0;
      } else if (m.type === 'stop') {
        this.fn = null;
      }
    };
  }
  process(inputs, outputs) {
    const out = outputs[0];
    const ch0 = out && out[0];
    if (!this.fn || !ch0 || ch0.length === 0) return true;
    const n = ch0.length;
    const step = this.songRate / sampleRate;
    for (let i = 0; i < n; i++) {
      this.acc += step;
      if (this.acc >= 1) {
        this.acc -= 1;
        this.t += 1;
      }
      let v;
      try {
        v = this.mode === 3 ? this.fn(this.t / this.songRate, this.songRate) : this.fn(this.t);
      } catch (e) {
        v = 0;
      }
      v = transform(this.mode, v);
      ch0[i] = v;
      for (let c = 1; c < out.length; c++) out[c][i] = v;
    }
    return true;
  }
}

registerProcessor('bytebeat', ByteBeatProcessor);
