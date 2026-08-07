from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st
from groq import APIStatusError

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from app_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"





def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).

    Real Groq API errors (rate limits, bad requests, etc.) are re-raised immediately
    rather than silently retried — retrying a rate-limit error 3x back-to-back just
    burns more quota for no benefit. The retry-on-Exception behavior below is only
    meant to fall back across different LangGraph stream_mode support, not to paper
    over genuine API failures.
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except APIStatusError:
        raise
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except APIStatusError:
        raise
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0e1117; }

    /* Hero header */
    .bwa-hero { padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid rgba(250,250,250,0.08); margin-bottom: 1.25rem; }
    .bwa-hero h1 {
        font-size: 2.1rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa 60%, #f0abfc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .bwa-hero p { color: rgba(250,250,250,0.55); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Sidebar sections */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(250,250,250,0.08); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { letter-spacing: 0.02em; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px; font-weight: 600; transition: transform 0.05s ease-in-out;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6); border: none;
    }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Dataframe / containers */
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Badge chips used in the Plan tab */
    .bwa-chip {
        display: inline-block; padding: 0.18rem 0.65rem; margin: 0 0.4rem 0.4rem 0;
        border-radius: 999px; font-size: 0.82rem; font-weight: 600;
        background: rgba(139,92,246,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bwa-hero">
        <h1>📝 Blog Writing Agent</h1>
        <p>Plan → research → draft, sections in parallel, powered by your local LangGraph pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("✍️ Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
        placeholder="e.g. State of Multimodal LLMs in 2026",
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("🗂️ Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "final": md_text,      # markdown body
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "final": "",
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"➡️ Node: `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                    "evidence_count": len(current_state.get("evidence", []) or []),
                    "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                    "sections_done": len(current_state.get("sections", []) or []),
                }
                progress_area.json(summary)

                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                out = payload
                st.session_state["last_out"] = out
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")

    except APIStatusError as e:
        status.update(label="❌ Stopped", state="error", expanded=False)
        msg = str(e)
        wait_match = re.search(r"try again in ([\d.]+)s", msg) or re.search(r"try again in (\d+)m([\d.]+)s", msg)
        code = getattr(e, "status_code", None)

        if code == 429:
            retry_hint = ""
            m = re.search(r"try again in (\d+)m([\d.]+)s", msg)
            if m:
                retry_hint = f" (Groq suggests waiting about {m.group(1)}m {int(float(m.group(2)))}s)"
            else:
                m2 = re.search(r"try again in ([\d.]+)s", msg)
                if m2:
                    retry_hint = f" (Groq suggests waiting about {int(float(m2.group(1)))}s)"
            st.error(
                f"⏳ Groq rate/quota limit reached{retry_hint}. "
                "This is a Groq account limit, not a bug in the app — wait and try again, "
                "or switch to a lighter model (e.g. `llama-3.1-8b-instant`) if this happens often.",
                icon="⏳",
            )
        elif code == 400:
            st.error(
                "⚠️ The model's response didn't match the expected format for this step "
                "(a structured-output call failed). Try again — this is usually transient.",
                icon="⚠️",
            )
        else:
            st.error(f"❌ Groq API error ({code}): {msg[:300]}", icon="❌")

        log(f"[error] Groq API error: {msg}")

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### {plan_dict.get('blog_title', '')}")
            st.markdown(
                f'<span class="bwa-chip">👥 {plan_dict.get("audience", "—")}</span>'
                f'<span class="bwa-chip">🎙️ {plan_dict.get("tone", "—")}</span>'
                f'<span class="bwa-chip">🏷️ {plan_dict.get("blog_kind", "—")}</span>',
                unsafe_allow_html=True,
            )

            tasks = plan_dict.get("tasks", [])
            if tasks:
                total_words = sum((t.get("target_words") or 0) for t in tasks)
                m1, m2, m3 = st.columns(3)
                m1.metric("Sections", len(tasks))
                m2.metric("Target words", f"{total_words:,}")
                m3.metric(
                    "Needs citations",
                    sum(1 for t in tasks if t.get("requires_citations")),
                )

                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            st.markdown(final_md, unsafe_allow_html=False)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("👋 Enter a topic in the sidebar and click **🚀 Generate Blog** to get started.")