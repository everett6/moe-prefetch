"""
Real prompts, fetched from where real people wrote them.

corpus_prompts.py built the old corpus out of my own templates crossed with my
own topic lists. It is varied in the way a crossword is varied -- every cell
filled by the same hand. Expert routing is a function of the hidden state, the
hidden state is a function of the text, and the text was mine, so every routing
statistic in this repo so far describes prompts that no user would ever send.

This fetches text humans wrote for their own reasons, in the two registers the
work actually targets:

  code        real GitHub issues (SWE-bench and live repos), crowd-sourced
              programming tasks (MBPP), hand-written problems (HumanEval),
              real Stack Overflow questions
  decision    real architecture and trade-off questions from Stack Exchange,
              real human first turns from OpenAssistant, human-written
              instructions from no_robots

Two properties the old corpus could not have:

  provenance  every record carries its source, a stable group key, the upstream
              URL, the licence and the fetch timestamp. A number measured on
              this corpus can be traced back to the human who wrote the prompt.
  groups      the split unit is the GROUP -- a repo, a Stack Exchange site, a
              no_robots category -- not the prompt. A held-out SWE-bench repo is
              unseen code in an unseen project. The old split held out one half
              of a template from the other half of the same template, which is
              a much easier question than the deployed model faces.

Fetching is cached and resumable: raw API responses land in data/prompts/raw/
keyed by request, so a re-run costs nothing and a rate-limit stall can be picked
up where it stopped. Stack Exchange anonymous quota is 300 requests/day/IP, so
the raw cache is not a nicety here.
"""
import hashlib
import html
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "data", "prompts")
RAWDIR = os.path.join(OUTDIR, "raw")
PROMPT_SET_VERSION = "real-v1-20260919"

# Real coding issues run to tens of thousands of characters. Long context is the
# point -- the old corpus had none -- but a prompt has to fit the capture n_ctx
# with room for the continuation, so the tail is cut and the cut is recorded.
MAX_CHARS = 20000

UA = {"User-Agent": "moe-prefetch-corpus/1.0 (research; local)"}


# -- plumbing ---------------------------------------------------------------
def _raw_path(key):
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return os.path.join(RAWDIR, f"{h}.json")


def http_json(url, params=None, headers=None, tries=5):
    """GET with an on-disk cache keyed by the full request, and backoff.

    The cache is keyed by URL+params rather than by call site, so re-running the
    fetcher after adding one source does not re-spend quota on the others.
    """
    key = url + "?" + urllib.parse.urlencode(sorted((params or {}).items()))
    path = _raw_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    delay = 2.0
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})},
                             timeout=90)
            if r.status_code == 200:
                data = r.json()
                os.makedirs(RAWDIR, exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
                return data
            if r.status_code in (429, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:300]}")
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"gave up on {url}")


def hf_rows(dataset, config, split, offset, length=100, where=None):
    """One page of a HF dataset. /filter when a predicate is given, /rows when not.

    Paging at 100 rows is the server's cap, not a choice.
    """
    base = "https://datasets-server.huggingface.co/"
    params = {"dataset": dataset, "config": config, "split": split,
              "offset": offset, "length": length}
    if where:
        params["where"] = where
        return http_json(base + "filter", params)
    return http_json(base + "rows", params)


def strip_html(s):
    """Stack Exchange bodies are HTML. Keep code, drop markup.

    Code blocks are the substantive half of a real programming question, so they
    are turned into fences rather than flattened into prose -- a model sees a
    very different prompt if the code loses its block structure.
    """
    s = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>",
               lambda m: "\n```\n" + html.unescape(m.group(1)) + "\n```\n",
               s, flags=re.S)
    s = re.sub(r"<code[^>]*>(.*?)</code>", lambda m: "`" + html.unescape(m.group(1)) + "`",
               s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(s)).strip()


def record(source, group, domain, text, url, licence):
    text = (text or "").strip()
    if not text:
        return None
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]
    return {
        "source": source, "group": group, "domain": domain,
        "text": text, "chars": len(text), "truncated": truncated,
        "url": url, "license": licence,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


# -- sources ----------------------------------------------------------------
def src_swebench(limit=1200):
    """Real GitHub issues from 12 real Python repos. group = repo."""
    out, offset = [], 0
    while len(out) < limit:
        page = hf_rows("princeton-nlp/SWE-bench", "default", "test", offset)
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            rec = record("swebench", row["repo"], "code", row["problem_statement"],
                         f"https://github.com/{row['repo']}/issues", "CC BY 4.0")
            if rec:
                rec["instance_id"] = row["instance_id"]
                out.append(rec)
        offset += len(rows)
        if offset >= page.get("num_rows_total", 0):
            break
    return out[:limit]


def src_mbpp(limit=1000):
    """All four splits: MBPP's own train/test/validation division is for code
    evaluation and means nothing to routing, and train alone is only 374 tasks."""
    out = []
    for split in ("train", "test", "validation", "prompt"):
        offset = 0
        while True:
            page = hf_rows("google-research-datasets/mbpp", "full", split, offset)
            rows = page.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r["row"]
                out.append(record("mbpp", "mbpp", "code", row["text"],
                                  "https://huggingface.co/datasets/google-research-datasets/mbpp",
                                  "CC BY 4.0"))
            offset += len(rows)
            if offset >= page.get("num_rows_total", 0):
                break
    return [r for r in out if r][:limit]


def src_humaneval(limit=164):
    out, offset = [], 0
    while len(out) < limit:
        page = hf_rows("openai/openai_humaneval", "openai_humaneval", "test", offset)
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            out.append(record("humaneval", "humaneval", "code",
                              "Complete this Python function:\n\n```python\n"
                              + row["prompt"] + "\n```",
                              "https://huggingface.co/datasets/openai/openai_humaneval",
                              "MIT"))
        offset += len(rows)
        if offset >= page.get("num_rows_total", 0):
            break
    return [r for r in out if r][:limit]


def src_no_robots(limit=800):
    """Human-written instructions. group = the dataset's own category."""
    out, offset = [], 0
    while len(out) < limit * 3:
        page = hf_rows("HuggingFaceH4/no_robots", "default", "train", offset)
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            cat = row.get("category", "unknown")
            msgs = row.get("messages") or []
            first = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            domain = "code" if cat == "Coding" else "decision"
            if cat in ("Coding", "Brainstorm", "Open QA", "Generation", "Classify"):
                out.append(record("no_robots", f"no_robots:{cat}", domain, first,
                                  "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
                                  "CC BY-NC 4.0"))
        offset += len(rows)
        if offset >= page.get("num_rows_total", 0):
            break
    return [r for r in out if r][:limit]


def src_oasst1(limit=700):
    """Real human first turns. Roots only: a reply is shaped by the model's
    previous answer, a root is what a person arrived with."""
    # /filter server-side, because roots are a small fraction of 84k messages and
    # paging /rows to find them costs ~400 requests for the same result.
    where = "\"role\"='prompter' AND \"lang\"='en'"
    out, offset = [], 0
    while len(out) < limit and offset < 40000:
        page = hf_rows("OpenAssistant/oasst1", "default", "train", offset, where=where)
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            if (row.get("role") == "prompter" and row.get("parent_id") in (None, "")
                    and row.get("lang") == "en"):
                out.append(record("oasst1", "oasst1", "decision", row.get("text"),
                                  "https://huggingface.co/datasets/OpenAssistant/oasst1",
                                  "Apache-2.0"))
        offset += len(rows)
        if offset >= page.get("num_rows_total", 0):
            break
    return [r for r in out if r][:limit]


def src_stackexchange(site, domain, pages=6, limit=400, tagged=None):
    """Real questions, most-voted first. group = the site, which is the community."""
    out = []
    for page in range(1, pages + 1):
        params = {"order": "desc", "sort": "votes", "site": site,
                  "filter": "withbody", "pagesize": 100, "page": page}
        if tagged:
            params["tagged"] = tagged
        data = http_json("https://api.stackexchange.com/2.3/questions", params)
        for it in data.get("items", []):
            body = strip_html(it.get("body", ""))
            text = html.unescape(it.get("title", "")) + "\n\n" + body
            g = f"se:{site}" + (f":{tagged}" if tagged else "")
            out.append(record(f"se_{site}", g, domain, text, it.get("link", ""),
                              "CC BY-SA 4.0"))
        if not data.get("has_more"):
            break
        time.sleep(0.3)
    return [r for r in out if r][:limit]


def src_github_issues(repo, pages=3, limit=250):
    """Live issues from a real repo. Pull requests are filtered out -- the API
    returns them through the issues endpoint, and a PR body is a different kind
    of text from a bug report."""
    out = []
    for page in range(1, pages + 1):
        key = f"gh:{repo}:{page}"
        path = _raw_path(key)
        if os.path.exists(path):
            with open(path) as f:
                items = json.load(f)
        else:
            cmd = ["gh", "api", "-X", "GET",
                   f"repos/{repo}/issues",
                   "-f", "state=all", "-f", "per_page=100", "-f", f"page={page}"]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if p.returncode != 0:
                print(f"  gh failed for {repo} p{page}: {p.stderr[:200]}", file=sys.stderr)
                break
            items = json.loads(p.stdout)
            os.makedirs(RAWDIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(items, f)
        if not items:
            break
        for it in items:
            if "pull_request" in it:
                continue
            text = (it.get("title") or "") + "\n\n" + (it.get("body") or "")
            out.append(record("github_issues", repo, "code", text,
                              it.get("html_url", ""), "repo terms"))
    return [r for r in out if r][:limit]


SOURCES = {
    "swebench":      lambda: src_swebench(2294),
    "mbpp":          lambda: src_mbpp(1000),
    "humaneval":     lambda: src_humaneval(164),
    # The issues endpoint returns pull requests too and most of these repos are
    # PR-heavy, so pages are generous relative to the issues they yield.
    "github_issues": lambda: (src_github_issues("ggml-org/llama.cpp", 12, 500)
                              + src_github_issues("vllm-project/vllm", 8, 300)
                              + src_github_issues("huggingface/transformers", 8, 300)
                              + src_github_issues("pytorch/pytorch", 8, 300)
                              + src_github_issues("numpy/numpy", 6, 200)),
    "so":            lambda: (src_stackexchange("stackoverflow", "code", 3, 200, tagged="python")
                              + src_stackexchange("stackoverflow", "code", 2, 120, tagged="c++")),
    "se_decision":   lambda: (src_stackexchange("softwareengineering", "decision", 4, 350)
                              + src_stackexchange("devops", "decision", 2, 120)
                              + src_stackexchange("security", "decision", 2, 120)),
    "no_robots":     lambda: src_no_robots(800),
    "oasst1":        lambda: src_oasst1(700),
}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    only = sys.argv[1:] or list(SOURCES)
    manifest_path = os.path.join(OUTDIR, "manifest.json")
    manifest = {"version": PROMPT_SET_VERSION, "max_chars": MAX_CHARS, "sources": {}}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        manifest.setdefault("sources", {})

    for name in only:
        out_path = os.path.join(OUTDIR, f"{name}.jsonl")
        if os.path.exists(out_path) and name in manifest["sources"]:
            print(f"{name}: already fetched ({manifest['sources'][name]['n']})")
            continue
        print(f"{name}: fetching ...", flush=True)
        t0 = time.time()
        recs = [r for r in SOURCES[name]() if r]
        # Dedupe within a source by text hash: Stack Exchange paging can repeat a
        # question when the vote order shifts mid-fetch.
        seen, uniq = set(), []
        for r in recs:
            if r["sha256"] in seen:
                continue
            seen.add(r["sha256"])
            r["id"] = f"{name}:{len(uniq):05d}"
            uniq.append(r)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            for r in uniq:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, out_path)
        h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        groups = sorted({r["group"] for r in uniq})
        manifest["sources"][name] = {
            "n": len(uniq), "file": f"{name}.jsonl", "sha256": h,
            "groups": groups, "n_groups": len(groups),
            "chars_total": sum(r["chars"] for r in uniq),
            "truncated": sum(r["truncated"] for r in uniq),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
        }
        with open(manifest_path + ".tmp", "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(manifest_path + ".tmp", manifest_path)
        print(f"  {len(uniq)} prompts, {len(groups)} groups, "
              f"{sum(r['chars'] for r in uniq)/1e6:.2f} M chars, "
              f"{time.time()-t0:.1f}s")

    tot = sum(v["n"] for v in manifest["sources"].values())
    print(f"\ntotal: {tot} prompts across {len(manifest['sources'])} sources")


if __name__ == "__main__":
    main()


# -- corpus for capture -----------------------------------------------------
def build_real_corpus(n=1400, seed=20260919):
    """Real prompts for capture, as (group, text, source) -- the same contract
    corpus_prompts.build_corpus offers, so capture_corpus can take either.

    The group key carries the domain as a prefix ("code/..." or "decision/...")
    so the split can stratify by domain without a second lookup, and a held-out
    group is a whole repo, site or category rather than a slice of one.

    Prompts are interleaved round-robin across groups. Capture is capped by a
    byte budget and may stop early; interleaving means any prefix of the corpus
    is still balanced across every group, rather than being all of Django and
    none of Stack Exchange.
    """
    rng = random.Random(seed)
    by_group = {}
    for fn in sorted(os.listdir(OUTDIR)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(OUTDIR, fn)) as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['domain']}/{r['group']}"
                by_group.setdefault(key, []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["sha256"])   # deterministic, not file order
        rng.shuffle(by_group[g])

    out, groups = [], sorted(by_group)
    i = 0
    while len(out) < n and any(by_group[g] for g in groups):
        g = groups[i % len(groups)]
        i += 1
        if by_group[g]:
            r = by_group[g].pop()
            out.append((g, r["text"], r["source"]))
    return out

