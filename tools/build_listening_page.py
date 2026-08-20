"""Build the blind A/B listening page for the M2 speaker-likeness condition.

The plan settles condition 3 by listening, not by a similarity score, and asks for the
comparison to be blind. So the page hides which system is which until a judgement is
recorded, and it keeps the objective numbers out of sight until then too - seeing that one
side already won on CER would colour the very judgement the plan wants independent.

The page is written next to the audio and opened from the filesystem. It is never
published: DATA_CREDITS.md lists generated audio as non-public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEMS = ("A_base", "B_adapted")


def blind_order(sentence_id: str, *, seed: int) -> tuple[str, str]:
    """Deterministic per-sentence play order, so neither system is always first."""
    digest = hashlib.sha256(f"{seed}:{sentence_id}".encode()).digest()
    return SYSTEMS if digest[0] % 2 == 0 else (SYSTEMS[1], SYSTEMS[0])


def tally_judgements(judgements: dict[str, str]) -> dict[str, int]:
    """Count how each system fared once the listener has decided."""
    tally = {"A_base": 0, "B_adapted": 0, "tie": 0, "total": 0}
    for choice in judgements.values():
        if choice in tally:
            tally[choice] += 1
        tally["total"] += 1
    return tally


def _rows(listening_dir: Path, seed: int) -> list[dict[str, Any]]:
    index = json.loads((listening_dir / "index.json").read_text(encoding="utf-8"))
    rows = []
    for pair in index["pairs"]:
        first, second = blind_order(pair["id"], seed=seed)
        rows.append(
            {
                "id": pair["id"],
                "text": pair["text"],
                "tags": pair.get("tags", []),
                "first": {"system": first, "file": pair[first]},
                "second": {"system": second, "file": pair[second]},
            }
        )
    return rows


def _metrics(project_root: Path) -> dict[str, Any]:
    m2 = project_root / "data/experiments/tsukuyomi_ojousama/m2"
    out: dict[str, Any] = {}
    try:
        similarity = json.loads((m2 / "speaker-similarity.json").read_text(encoding="utf-8"))
        out["similarity"] = {
            name: payload["per_file"] for name, payload in similarity["systems"].items()
        }
        out["similarity_summary"] = {
            name: payload["summary"] for name, payload in similarity["systems"].items()
        }
        out["paired"] = similarity.get("paired_comparison", {})
    except FileNotFoundError:
        pass
    for label, filename in (("A_base", "zeroshot"), ("B_adapted", "t1")):
        try:
            report = json.loads(
                (m2 / f"{filename}-intelligibility.json").read_text(encoding="utf-8")
            )
            out.setdefault("cer", {})[label] = {
                row["id"]: row["cer"] for row in report["sentences"]
            }
            out.setdefault("cer_summary", {})[label] = report["summary"]
        except FileNotFoundError:
            pass
    return out


def render(rows: list[dict[str, Any]], metrics: dict[str, Any], reference: str) -> str:
    payload = json.dumps(
        {"rows": rows, "metrics": metrics, "reference": reference}, ensure_ascii=False
    )
    return _TEMPLATE.replace("__PAYLOAD__", payload)


_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M2 ブラインド聴取</title>
<style>
  :root {
    --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66; --line: #e3e3df;
    --card: #ffffff; --accent: #2f6f4f; --accent-weak: #e8f1ec; --warn: #8a5a2b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17171a; --fg: #ececea; --muted: #9a9a95; --line: #2e2e33;
      --card: #1f1f23; --accent: #7fc4a0; --accent-weak: #22322a; --warn: #d2a273;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 96px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: .01em; }
  .sub { color: var(--muted); margin: 0 0 28px; font-size: 14px; }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 22px; }
  .panel h2 { font-size: 15px; margin: 0 0 10px; }
  .panel p { margin: 0 0 8px; color: var(--muted); font-size: 14px; }
  audio { width: 100%; height: 34px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 16px 18px; margin-bottom: 14px; }
  .card.done { border-color: var(--accent); }
  .head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .sid { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; }
  .text { margin: 6px 0 12px; font-size: 16px; }
  .tags { color: var(--muted); font-size: 12px; }
  .players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  @media (max-width: 620px) { .players { grid-template-columns: 1fr; } }
  .slot label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
  .choices { display: flex; flex-wrap: wrap; gap: 8px; }
  button.choice { font: inherit; font-size: 14px; padding: 7px 14px; border-radius: 999px;
    border: 1px solid var(--line); background: transparent; color: var(--fg); cursor: pointer; }
  button.choice:hover { border-color: var(--accent); }
  button.choice[aria-pressed="true"] { background: var(--accent-weak); border-color: var(--accent);
    color: var(--fg); font-weight: 600; }
  .reveal { margin-top: 10px; font-size: 13px; color: var(--muted); }
  .reveal b { color: var(--fg); }
  .bar { position: fixed; left: 0; right: 0; bottom: 0; background: var(--card);
    border-top: 1px solid var(--line); padding: 12px 20px; }
  .bar .inner { max-width: 860px; margin: 0 auto; display: flex; align-items: center;
    justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .count { font-variant-numeric: tabular-nums; }
  .actions { display: flex; gap: 8px; }
  .actions button { font: inherit; font-size: 14px; padding: 8px 16px; border-radius: 8px;
    border: 1px solid var(--line); background: transparent; color: var(--fg); cursor: pointer; }
  .actions button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .actions button:disabled { opacity: .45; cursor: default; }
  pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
    padding: 14px; overflow-x: auto; font-size: 12.5px; white-space: pre-wrap; }
  .note { color: var(--warn); font-size: 13px; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 6px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="wrap">
  <h1>M2 ブラインド聴取</h1>
  <p class="sub">完了条件3「話者らしさが base TTS より主観評価で改善している」の判定用。どちらがどの系かは、選択するまで伏せています。</p>

  <div class="panel">
    <h2>基準となる本人の声</h2>
    <p>まずこれを聴いて、つくよみちゃんの声を掴んでください。学習にも生成にも使っていない実際の収録です。</p>
    <audio controls preload="none" id="ref"></audio>
  </div>

  <div class="panel">
    <h2>聴き方</h2>
    <p>各文で「1」と「2」を聴き比べ、<b>基準の声にどちらが近いか</b>だけを選んでください。明瞭さや読み方の自然さではなく、声質（音高・響き・話速・抑揚）で判断します。全部でなくても、10文ほどで傾向は掴めます。</p>
    <p class="note">客観指標は判定を歪めないよう、選択後に各文で表示します。</p>
  </div>

  <div id="list"></div>
</div>

<div class="bar"><div class="inner">
  <span class="count" id="count">0 / 0 判定済み</span>
  <div class="actions">
    <button id="summary">集計を見る</button>
    <button id="copy" class="primary">結果をコピー</button>
  </div>
</div></div>

<script>
const DATA = __PAYLOAD__;
const store = {};
const list = document.getElementById('list');
document.getElementById('ref').src = DATA.reference;

const LABEL = { A_base: 'A: zero-shot（学習なし）', B_adapted: 'B: speaker inversion（適応後）' };

function metricsFor(id, system) {
  const bits = [];
  const cer = DATA.metrics.cer && DATA.metrics.cer[system] && DATA.metrics.cer[system][id];
  if (cer !== undefined) bits.push('CER ' + cer.toFixed(3));
  const sims = DATA.metrics.similarity && DATA.metrics.similarity[
    system === 'A_base' ? 'T0_zero_shot' : 'T1_speaker_inversion'];
  if (sims && sims[id] !== undefined) bits.push('類似度 ' + sims[id].toFixed(3));
  return bits.join(' / ');
}

DATA.rows.forEach((row, index) => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="head"><span class="sid">${index + 1} / ${DATA.rows.length} · ${row.id}</span>
      <span class="tags">${row.tags.join(' · ')}</span></div>
    <div class="text">${row.text}</div>
    <div class="players">
      <div class="slot"><label>1</label><audio controls preload="none" src="${row.first.file}"></audio></div>
      <div class="slot"><label>2</label><audio controls preload="none" src="${row.second.file}"></audio></div>
    </div>
    <div class="choices">
      <button class="choice" data-pick="first">1 が近い</button>
      <button class="choice" data-pick="second">2 が近い</button>
      <button class="choice" data-pick="tie">差がない</button>
    </div>
    <div class="reveal" hidden></div>`;

  const reveal = card.querySelector('.reveal');
  card.querySelectorAll('.choice').forEach(button => {
    button.addEventListener('click', () => {
      card.querySelectorAll('.choice').forEach(b => b.setAttribute('aria-pressed', 'false'));
      button.setAttribute('aria-pressed', 'true');
      const pick = button.dataset.pick;
      store[row.id] = pick === 'tie' ? 'tie' : row[pick].system;
      card.classList.add('done');
      reveal.hidden = false;
      reveal.innerHTML =
        `1 = <b>${LABEL[row.first.system]}</b> ${metricsFor(row.id, row.first.system)}<br>` +
        `2 = <b>${LABEL[row.second.system]}</b> ${metricsFor(row.id, row.second.system)}`;
      updateCount();
    });
  });
  list.appendChild(card);
});

function tally() {
  const t = { A_base: 0, B_adapted: 0, tie: 0 };
  Object.values(store).forEach(v => { if (v in t) t[v] += 1; });
  return t;
}

function updateCount() {
  document.getElementById('count').textContent =
    `${Object.keys(store).length} / ${DATA.rows.length} 判定済み`;
}

document.getElementById('summary').addEventListener('click', () => {
  const t = tally();
  const judged = Object.keys(store).length;
  const s = DATA.metrics.similarity_summary || {};
  const c = DATA.metrics.cer_summary || {};
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.innerHTML = `<h2>集計</h2>
    <table>
      <tr><th>判定</th><th class="num">件数</th></tr>
      <tr><td>B: speaker inversion が近い</td><td class="num">${t.B_adapted}</td></tr>
      <tr><td>A: zero-shot が近い</td><td class="num">${t.A_base}</td></tr>
      <tr><td>差がない</td><td class="num">${t.tie}</td></tr>
      <tr><td>判定済み</td><td class="num">${judged} / ${DATA.rows.length}</td></tr>
    </table>
    <h2 style="margin-top:18px">客観指標（参考）</h2>
    <table>
      <tr><th></th><th class="num">明瞭</th><th class="num">平均CER</th><th class="num">類似度 mean</th></tr>
      <tr><td>A: zero-shot</td>
        <td class="num">${c.A_base ? c.A_base.intelligible + '/' + c.A_base.total : '-'}</td>
        <td class="num">${c.A_base ? c.A_base.mean_cer.toFixed(4) : '-'}</td>
        <td class="num">${s.T0_zero_shot ? s.T0_zero_shot.mean.toFixed(4) : '-'}</td></tr>
      <tr><td>B: speaker inversion</td>
        <td class="num">${c.B_adapted ? c.B_adapted.intelligible + '/' + c.B_adapted.total : '-'}</td>
        <td class="num">${c.B_adapted ? c.B_adapted.mean_cer.toFixed(4) : '-'}</td>
        <td class="num">${s.T1_speaker_inversion ? s.T1_speaker_inversion.mean.toFixed(4) : '-'}</td></tr>
    </table>`;
  document.querySelector('.wrap').appendChild(panel);
  panel.scrollIntoView({ behavior: 'smooth' });
});

document.getElementById('copy').addEventListener('click', async () => {
  const t = tally();
  const out = {
    milestone: 'M2', condition: '3_speaker_likeness_beats_base_tts',
    judged: Object.keys(store).length, total: DATA.rows.length,
    tally: t, per_sentence: store,
  };
  const text = JSON.stringify(out, null, 2);
  try { await navigator.clipboard.writeText(text); } catch (e) { /* fall through */ }
  const pre = document.createElement('pre');
  pre.textContent = text;
  document.querySelector('.wrap').appendChild(pre);
  pre.scrollIntoView({ behavior: 'smooth' });
});

updateCount();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the M2 blind listening page")
    parser.add_argument(
        "--listening-dir",
        type=Path,
        default=Path("data/experiments/tsukuyomi_ojousama/m2/listening"),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    listening_dir = args.listening_dir
    output = args.output or listening_dir / "compare.html"
    rows = _rows(listening_dir, args.seed)
    metrics = _metrics(Path.cwd())
    reference = next((p.name for p in sorted(listening_dir.glob("_reference_natural_*.wav"))), "")
    output.write_text(render(rows, metrics, reference), encoding="utf-8")
    print(f"{output} ({output.stat().st_size} bytes, {len(rows)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
