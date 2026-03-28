/**
 * MiniGPT Web Worker
 * Runs ONNX inference off the main thread so the UI stays responsive.
 *
 * Messages IN  (from main thread):
 *   { type: 'init',     modelUrl: string, tokenizerUrl: string }
 *   { type: 'generate', prompt: string, maxTokens: number, temperature: number, topK: number }
 *   { type: 'cancel' }
 *
 * Messages OUT (to main thread):
 *   { type: 'ready' }
 *   { type: 'token',    text: string, tokenId: number }
 *   { type: 'done',     fullText: string, tps: number, totalTokens: number }
 *   { type: 'error',    message: string }
 *   { type: 'progress', loaded: number, total: number }
 */

// ─── ONNX Runtime Web ─────────────────────────────────────────────────────────
// Load the JS bundle from CDN
importScripts('https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/ort.min.js');

// CRITICAL: Tell ORT where its WASM files live.
// Without this it tries to load them from the local server (which doesn't have them)
// and the browser spams 404 errors for ort-wasm-simd-threaded.wasm etc.
ort.env.wasm.wasmPaths = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/';

let session   = null;
let tokenizer = null;
let cancelled = false;

// ─── Model Config (must match Python MiniGPTConfig) ────────────────────────
const MODEL_CONFIG = {
  n_head:       4,
  n_embd:       128,
  head_dim:     32,    // n_embd / n_head
  n_layer:      4,
  vocab_size:   5000,
  block_size:   256,
  eos_token_id: 2,     // </s>
};

// ─── Tokenizer ────────────────────────────────────────────────────────────────

// Full ByteLevel BPE unicode → byte mapping (matches HuggingFace tokenizers library).
// ByteLevel BPE encodes every byte as a unicode character to avoid unknown tokens.
// Control chars and whitespace are remapped starting at U+0100.
// Without this map, characters like Ċ (newline) and ĉ (tab) render as literal glyphs.
function buildByteDecoder() {
  // Printable ASCII + Latin-1 supplement map to themselves
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);   // ! to ~
  for (let i = 161; i <= 172; i++) bs.push(i);  // ¡ to ¬
  for (let i = 174; i <= 255; i++) bs.push(i);  // ® to ÿ

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }

  // Returns Map<unicodeChar, byte>
  const decoder = new Map();
  for (let i = 0; i < bs.length; i++) decoder.set(String.fromCharCode(cs[i]), bs[i]);
  return decoder;
}
const BYTE_DECODER = buildByteDecoder();

function byteLevelDecode(token) {
  // Convert a BPE token string back to the original UTF-8 text
  const bytes = [];
  for (const ch of token) {
    const b = BYTE_DECODER.get(ch);
    if (b !== undefined) bytes.push(b);
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
}

class SimpleBPETokenizer {
  constructor(vocab, merges) {
    this.vocab     = vocab;
    this.idToToken = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
    this.merges    = merges;
    this.bosId     = vocab['<s>']   ?? 0;
    this.eosId     = vocab['</s>'] ?? 2;
    this.unkId     = vocab['<unk>'] ?? 3;
  }

  encode(text) {
    const tokens = [this.bosId];
    for (const char of text) {
      const id = this.vocab['Ġ' + char] ?? this.vocab[char] ?? this.unkId;
      tokens.push(id);
    }
    return tokens;
  }

  decode(ids) {
    // Join all token strings then run the full byte-level decode in one pass.
    // This correctly handles multi-byte UTF-8 sequences split across tokens.
    const tokenStr = ids
      .filter(id => id !== this.bosId && id !== this.eosId)
      .map(id => this.idToToken[id] ?? '')
      .join('');
    return byteLevelDecode(tokenStr);
  }
}

// ─── KV Cache ─────────────────────────────────────────────────────────────────
function createEmptyKVCache() {
  const cache = {};
  const { n_layer, n_head, head_dim } = MODEL_CONFIG;
  for (let i = 0; i < n_layer; i++) {
    cache[`past_k_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, n_head, 0, head_dim]);
    cache[`past_v_${i}`] = new ort.Tensor('float32', new Float32Array(0), [1, n_head, 0, head_dim]);
  }
  return cache;
}

function extractPresentKVCache(results) {
  const cache = {};
  for (let i = 0; i < MODEL_CONFIG.n_layer; i++) {
    cache[`past_k_${i}`] = results[`present_k_${i}`];
    cache[`past_v_${i}`] = results[`present_v_${i}`];
  }
  return cache;
}

// ─── Sampling ─────────────────────────────────────────────────────────────────
function topKSample(logits, topK = 40, temperature = 0.8) {
  const n      = logits.length;
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

  const indexed = Array.from(scaled).map((v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const topKSet = new Set(indexed.slice(0, topK).map(([, i]) => i));

  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) if (topKSet.has(i) && scaled[i] > maxVal) maxVal = scaled[i];

  let sum = 0;
  const probs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (topKSet.has(i)) { probs[i] = Math.exp(scaled[i] - maxVal); sum += probs[i]; }
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;

  let r = Math.random();
  for (let i = 0; i < n; i++) { r -= probs[i]; if (r <= 0) return i; }
  return indexed[0][1];
}

// ─── Init ──────────────────────────────────────────────────────────────────────
async function init(modelUrl, tokenizerUrl) {
  try {
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 2, 4);
    ort.env.wasm.simd       = true;

    self.postMessage({ type: 'progress', phase: 'model', loaded: 0, total: 100 });

    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders:  ['wasm'],
      graphOptimizationLevel: 'all',
      intraOpNumThreads:   ort.env.wasm.numThreads,
    });

    self.postMessage({ type: 'progress', phase: 'model', loaded: 50, total: 100 });

    // Load tokenizer vocab
    const vocabResp  = await fetch(tokenizerUrl + '/vocab.json');
    const vocab      = await vocabResp.json();
    const mergesResp = await fetch(tokenizerUrl + '/merges.txt');
    const mergesText = await mergesResp.text();
    const merges     = mergesText.split('\n').filter(l => l && !l.startsWith('#'));
    tokenizer = new SimpleBPETokenizer(vocab, merges);

    self.postMessage({ type: 'progress', phase: 'tokenizer', loaded: 100, total: 100 });
    self.postMessage({ type: 'ready' });

  } catch (err) {
    self.postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
}

// ─── Generate ─────────────────────────────────────────────────────────────────
async function generate(prompt, maxTokens = 100, temperature = 0.8, topK = 40) {
  if (!session || !tokenizer) {
    self.postMessage({ type: 'error', message: 'Model not loaded' });
    return;
  }

  cancelled = false;
  const { eos_token_id, n_head, head_dim, vocab_size } = MODEL_CONFIG;

  let tokenIds  = tokenizer.encode(prompt);
  const t0      = performance.now();
  let generated = 0;
  let fullText  = prompt;

  // Step 1: process full prompt, build KV cache
  const promptTensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(tokenIds.map(BigInt)),
    [1, tokenIds.length]
  );
  let feeds   = { input_ids: promptTensor, ...createEmptyKVCache() };
  let results = await session.run(feeds);
  let pastKV  = extractPresentKVCache(results);

  // Step 2: autoregressive decode
  for (let i = 0; i < maxTokens; i++) {
    if (cancelled) break;

    const logitsData  = results.logits.data;
    const seqLen      = results.logits.dims[1];
    const lastLogits  = logitsData.slice((seqLen - 1) * vocab_size, seqLen * vocab_size);
    const nextTokenId = topKSample(lastLogits, topK, temperature);

    if (nextTokenId === eos_token_id) break;

    const tokenText = tokenizer.decode([nextTokenId]);
    fullText       += tokenText;
    generated++;

    self.postMessage({ type: 'token', text: tokenText, tokenId: nextTokenId });

    const nextTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(nextTokenId)]), [1, 1]);
    feeds   = { input_ids: nextTensor, ...pastKV };
    results = await session.run(feeds);
    pastKV  = extractPresentKVCache(results);
  }

  const elapsed = (performance.now() - t0) / 1000;
  const tps     = generated / elapsed;

  self.postMessage({
    type:      'done',
    fullText,
    tps:       Math.round(tps),
    totalTokens: generated,
    elapsedMs: Math.round(elapsed * 1000),
  });
}

// ─── Message Handler ───────────────────────────────────────────────────────────
self.onmessage = async (e) => {
  const { type, ...payload } = e.data;
  switch (type) {
    case 'init':     await init(payload.modelUrl, payload.tokenizerUrl); break;
    case 'generate': await generate(payload.prompt, payload.maxTokens, payload.temperature, payload.topK); break;
    case 'cancel':   cancelled = true; break;
  }
};
