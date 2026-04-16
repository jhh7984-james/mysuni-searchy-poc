require('dotenv').config();
const express = require('express');
const fs = require('fs');
const path = require('path');
const iconv = require('./node_modules/iconv-lite');

const app = express();
app.use(express.json());
app.use(express.static('public'));

// ─── Mode detection ────────────────────────────────────────────────────────────
// If ANTHROPIC_API_KEY is set → use Claude for semantic search (fast, great quality)
// Otherwise              → use local multilingual embeddings (slower first run, no API needed)
const USE_CLAUDE = !!process.env.ANTHROPIC_API_KEY;
console.log(`Search mode: ${USE_CLAUDE ? 'Claude API (intent re-ranking)' : 'Local embeddings (multilingual-e5-small)'}`);

// ─── CSV Parsing ───────────────────────────────────────────────────────────────
function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
      else inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function stripHtml(html) {
  if (!html) return '';
  return html
    .replace(/<[^>]*>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Data Loading ──────────────────────────────────────────────────────────────
let cards = [];
let isReady = false;
let embeddingStatus = 'pending'; // pending | building | ready | error

function loadCards() {
  console.log('Loading CardInformation.csv...');
  const buf = fs.readFileSync(path.join(__dirname, 'CardInformation.csv'));
  const content = iconv.decode(buf, 'cp949');

  const rawLines = [];
  let buffer = '';
  let inQuotes = false;
  for (let i = 0; i < content.length; i++) {
    const ch = content[i];
    if (ch === '"') { inQuotes = !inQuotes; buffer += ch; }
    else if ((ch === '\r' || ch === '\n') && !inQuotes) {
      if (ch === '\r' && content[i + 1] === '\n') i++;
      if (buffer.trim()) rawLines.push(buffer);
      buffer = '';
    } else { buffer += ch; }
  }
  if (buffer.trim()) rawLines.push(buffer);

  const headers = parseCSVLine(rawLines[0]);
  const idx = {
    id:          headers.indexOf('카드아이디'),
    name:        headers.indexOf('카드명(Ko)'),
    tagline:     headers.indexOf('카드 표시문구(Ko)'),
    intro:       headers.indexOf('Card 소개(ko)'),
    courseType:  headers.indexOf('과정유형'),
    cardType:    headers.indexOf('카드유형'),
    channel:     headers.indexOf('Channel'),
    learners:    headers.indexOf('학습자'),
    completers:  headers.indexOf('이수자'),
    regDate:     headers.indexOf('등록일자'),   // P열
    isPublic:    headers.indexOf('공개여부'),   // W열
    isActive:    headers.indexOf('사용여부'),   // X열
  };

  cards = rawLines.slice(1).map(line => {
    const cols = parseCSVLine(line);
    const name = (cols[idx.name] || '').trim();
    if (!name) return null;
    const intro = stripHtml(cols[idx.intro] || '');
    const tagline = (cols[idx.tagline] || '').trim();
    return {
      id:          (cols[idx.id] || '').trim(),
      name,
      tagline,
      intro,
      courseType:  (cols[idx.courseType] || '').trim(),
      cardType:    (cols[idx.cardType] || '').trim(),
      channel:     (cols[idx.channel] || '').trim(),
      learners:    parseInt(cols[idx.learners] || '0', 10) || 0,
      completers:  parseInt(cols[idx.completers] || '0', 10) || 0,
      regDate:     (cols[idx.regDate] || '').trim(),
      isPublic:    (cols[idx.isPublic] || '').trim(),
      isActive:    (cols[idx.isActive] || '').trim(),
      searchText:  `${name} ${tagline} ${intro}`.toLowerCase(),
    };
  }).filter(c => c !== null && c.isPublic === '공개' && c.isActive === 'Yes');

  console.log(`Loaded ${cards.length} cards.`);
  isReady = true;
}

// ─── BM25 Index ────────────────────────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase().split(/[\s,.!?;:()\[\]{}<>\/\\|'"·•\-]+/).filter(t => t.length >= 1);
}

let bm25Index = null;

function buildBM25Index() {
  console.log('Building BM25 index...');
  const N = cards.length;
  const df = {};
  const tfDocs = cards.map(card => {
    const tokens = tokenize(card.searchText);
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    return { tf, len: tokens.length };
  });
  for (const { tf } of tfDocs) {
    for (const term of Object.keys(tf)) df[term] = (df[term] || 0) + 1;
  }
  const avgLen = tfDocs.reduce((s, d) => s + d.len, 0) / N;
  const idf = {};
  for (const [term, freq] of Object.entries(df)) {
    idf[term] = Math.log((N - freq + 0.5) / (freq + 0.5) + 1);
  }
  bm25Index = { tfDocs, idf, avgLen };
  console.log('BM25 index ready.');
}

function bm25Score(queryTokens, docIdx, k1 = 1.5, b = 0.75) {
  const { tfDocs, idf, avgLen } = bm25Index;
  const { tf, len } = tfDocs[docIdx];
  let score = 0;
  for (const term of queryTokens) {
    if (!idf[term]) continue;
    const f = tf[term] || 0;
    score += idf[term] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len / avgLen));
  }
  return score;
}

function keywordSearch(query, topK = 10) {
  const keywords = query.toLowerCase().split(/\s+/).filter(k => k.length >= 1);
  if (keywords.length === 0) return [];

  const nameText   = card => card.name.toLowerCase();
  const bodyText   = card => `${card.tagline} ${card.intro}`.toLowerCase();

  const results = cards
    .map(card => {
      const inName = keywords.filter(kw => nameText(card).includes(kw)).length;
      const inBody = keywords.filter(kw => bodyText(card).includes(kw)).length;
      const score  = inName * 2 + inBody; // 제목 매칭 가중치 2배
      return { card, score };
    })
    .filter(x => x.score > 0)
    // 1순위: 제목 매칭 가중치(score) 높은 것, 2순위: 등록일 최신순
    .sort((a, b) => b.score - a.score || (b.card.regDate || '').localeCompare(a.card.regDate || ''));

  return results.slice(0, topK).map(x => x.card);
}

function keywordSearchWide(query, topK = 60) {
  if (!bm25Index) buildBM25Index();
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return cards.slice(0, topK);
  const scores = cards.map((_, i) => ({ i, score: bm25Score(queryTokens, i) })).filter(x => x.score > 0);
  scores.sort((a, b) => b.score - a.score);

  let candidates = scores.slice(0, topK).map(x => cards[x.i]);

  // Expand with per-word search if needed
  if (candidates.length < 15) {
    const seen = new Set(candidates.map(c => c.id));
    for (const word of queryTokens) {
      if (word.length < 2) continue;
      const extra = cards.map((_, i) => ({ i, score: bm25Score([word], i) })).filter(x => x.score > 0);
      extra.sort((a, b) => b.score - a.score);
      for (const x of extra.slice(0, 10)) {
        if (!seen.has(cards[x.i].id)) { candidates.push(cards[x.i]); seen.add(cards[x.i].id); }
      }
    }
  }

  // Last resort: high-learner popular courses
  if (candidates.length < 8) {
    const popular = [...cards].sort((a, b) => b.learners - a.learners).slice(0, 30);
    const seen = new Set(candidates.map(c => c.id));
    for (const c of popular) { if (!seen.has(c.id)) candidates.push(c); }
  }

  return candidates.slice(0, topK);
}

// ─── Semantic Search: Claude API mode ──────────────────────────────────────────
let anthropicClient = null;
if (USE_CLAUDE) {
  const Anthropic = require('@anthropic-ai/sdk');
  anthropicClient = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}

async function semanticSearchClaude(query, topK = 10) {
  const pool = keywordSearchWide(query, 30);

  const cardList = pool.map((c, i) =>
    `[${i + 1}] 제목: "${c.name}"${c.tagline ? ` | "${c.tagline}"` : ''}${c.intro ? ` | ${c.intro.slice(0, 100)}` : ''}`
  ).join('\n');

  const response = await anthropicClient.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 1000,
    system: 'MySUNI 기업 학습 플랫폼 과정 추천 AI. 학습자의 의도를 정확히 파악하여 관련 과정을 선별합니다.',
    messages: [{
      role: 'user',
      content: `검색 의도: "${query}"

아래 목록에서 이 의도와 가장 관련된 과정을 최대 ${topK}개 선택하세요. 키워드가 정확히 일치하지 않아도 학습 목적/맥락이 맞으면 포함하세요.

${cardList}

JSON 형식으로만 응답하세요:
{"results": [{"index": 번호, "reason": "한 문장 추천 이유"}]}`
    }],
  });

  const text = response.content[0].text.trim();
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('Claude 응답 파싱 실패');
  const parsed = JSON.parse(match[0]);
  return parsed.results
    .filter(r => r.index >= 1 && r.index <= pool.length)
    .slice(0, topK)
    .map(r => ({ ...pool[r.index - 1], reason: r.reason }));
}

// ─── Semantic Search: Local Embeddings mode ────────────────────────────────────
let embedder = null;
let cardEmbeddings = null;
const EMBED_CACHE = path.join(__dirname, 'embeddings_cache.json');

async function initLocalEmbedder() {
  embeddingStatus = 'building';
  console.log('Loading multilingual-e5-small model (first run downloads ~150MB)...');
  const { pipeline } = await import('@xenova/transformers');
  embedder = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small');
  console.log('Model loaded!');

  // Load or build embeddings cache
  if (fs.existsSync(EMBED_CACHE)) {
    console.log('Loading cached embeddings...');
    const cached = JSON.parse(fs.readFileSync(EMBED_CACHE, 'utf-8'));
    if (cached.count === cards.length) {
      cardEmbeddings = cached.embeddings.map(e => new Float32Array(e));
      embeddingStatus = 'ready';
      console.log(`Loaded ${cardEmbeddings.length} cached embeddings.`);
      return;
    }
    console.log('Cache mismatch, rebuilding...');
  }

  console.log(`Computing embeddings for ${cards.length} cards (this may take several minutes)...`);
  cardEmbeddings = [];
  const BATCH = 32;
  for (let i = 0; i < cards.length; i += BATCH) {
    const batch = cards.slice(i, i + BATCH);
    const texts = batch.map(c => `passage: ${c.name} ${c.tagline} ${c.intro.slice(0, 200)}`);
    const output = await embedder(texts, { pooling: 'mean', normalize: true });
    for (let j = 0; j < batch.length; j++) {
      cardEmbeddings.push(new Float32Array(output[j].data));
    }
    if ((i + BATCH) % 500 === 0 || i + BATCH >= cards.length) {
      process.stdout.write(`\r  Progress: ${Math.min(i + BATCH, cards.length)}/${cards.length}`);
    }
  }
  console.log('\nEmbeddings computed! Saving cache...');
  fs.writeFileSync(EMBED_CACHE, JSON.stringify({
    count: cards.length,
    embeddings: cardEmbeddings.map(e => Array.from(e)),
  }));
  embeddingStatus = 'ready';
  console.log('Cache saved. Semantic search ready.');
}

function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // already normalized
}

async function semanticSearchLocal(query, topK = 10) {
  if (embeddingStatus !== 'ready') {
    throw new Error('임베딩 인덱스 준비 중입니다. 잠시 후 다시 시도해주세요.');
  }
  const output = await embedder(`query: ${query}`, { pooling: 'mean', normalize: true });
  const queryEmb = new Float32Array(output.data);

  const scores = cardEmbeddings.map((emb, i) => ({ i, score: cosineSimilarity(queryEmb, emb) }));
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK).map(x => ({ ...cards[x.i], similarityScore: x.score }));
}

// ─── Unified Semantic Search ───────────────────────────────────────────────────
async function semanticSearch(query, topK = 10) {
  if (USE_CLAUDE) return semanticSearchClaude(query, topK);
  return semanticSearchLocal(query, topK);
}

// ─── Routes ────────────────────────────────────────────────────────────────────
app.get('/api/stats', (_req, res) => {
  res.json({
    cardCount: cards.length,
    ready: isReady,
    mode: USE_CLAUDE ? 'claude' : 'local',
    embeddingStatus: USE_CLAUDE ? 'n/a' : embeddingStatus,
  });
});

app.post('/api/keyword-search', (req, res) => {
  const { query } = req.body;
  if (!query?.trim()) return res.json({ results: [], time: 0 });
  const start = Date.now();
  try {
    const results = keywordSearch(query.trim(), 10);
    res.json({ results, time: Date.now() - start });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/semantic-search', async (req, res) => {
  const { query } = req.body;
  if (!query?.trim()) return res.json({ results: [], time: 0 });
  const start = Date.now();
  try {
    const results = await semanticSearch(query.trim(), 10);
    res.json({ results, time: Date.now() - start });
  } catch (e) {
    console.error('Semantic search error:', e.message);
    res.status(503).json({ error: e.message });
  }
});

// ─── Startup ───────────────────────────────────────────────────────────────────
loadCards();
buildBM25Index();

if (!USE_CLAUDE) {
  initLocalEmbedder().catch(e => {
    console.error('Embedding init failed:', e.message);
    embeddingStatus = 'error';
  });
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n Search POC → http://localhost:${PORT}`);
  if (!USE_CLAUDE) {
    console.log(' 의미 검색 모드: 로컬 임베딩 (백그라운드에서 초기화 중...)');
    console.log(' API 키를 사용하려면: .env 파일에 ANTHROPIC_API_KEY=sk-ant-api03-... 추가\n');
  } else {
    console.log(' 의미 검색 모드: Claude API\n');
  }
});
