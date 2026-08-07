from __future__ import annotations

import json
import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → Reducer)
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
# Plain prose generation (worker sections) — tight cap is fine and desirable for speed,
# since a truncated section is still usable text, just shorter than intended.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=900,
)

# Structured/JSON tool-calling outputs (router, research extractor, orchestrator planner) —
# these need a much bigger budget. Unlike plain prose, a truncated JSON tool call isn't
# just shorter, it's *invalid* — Groq can't parse partial JSON and throws tool_use_failed.
llm_structured = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=3000,
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm_structured.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    print(
        f"[router] mode={decision.mode!r} needs_research={decision.needs_research} "
        f"reason={decision.reason!r} queries={decision.queries}"
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        print("[research] TAVILY_API_KEY is not set — skipping search, evidence will be empty.")
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        print(f"[research] query={query!r} -> {len(results or [])} raw results")
        out: List[dict] = []
        for r in results or []:
            snippet = (r.get("content") or r.get("snippet") or "").strip()
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."
            out.append(
                {
                    "title": (r.get("title") or "")[:150],
                    "url": r.get("url") or "",
                    "snippet": snippet,
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as e:
        print(f"[research] query={query!r} FAILED: {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:6]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=4))

    # Hard cap total items sent to the extractor, regardless of how many
    # queries/results came back — this is what actually protects the TPM limit.
    MAX_RAW_ITEMS = 15
    if len(raw) > MAX_RAW_ITEMS:
        print(f"[research] trimming raw results from {len(raw)} to {MAX_RAW_ITEMS} to stay under token limits")
        raw = raw[:MAX_RAW_ITEMS]

    if not raw:
        print("[research] no raw results from any query -> evidence will be empty")
        return {"evidence": []}

    extractor = llm_structured.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{json.dumps(raw, ensure_ascii=False)}"
                )
            ),
        ]
    )
    print(f"[research] extractor produced {len(pack.evidence)} evidence items before dedup/filter")

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        before_filter = len(evidence)
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]
        print(f"[research] open_book recency filter: {before_filter} -> {len(evidence)} items (cutoff={cutoff})")

    print(f"[research] final evidence count: {len(evidence)}")
    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 3–5 tasks, each with goal + 3–5 bullets + target_words.
- Keep target_words modest: 150–300 words per section unless the topic truly needs more.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm_structured.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def _clean_section_heading(section_md: str, fallback_title: str) -> str:
    """
    Defensive cleanup for small-model prompt-following slips:
    - Strips a literal leaked "<Section Title>" placeholder if the model echoed it back.
    - Guarantees the section starts with a proper '## ' Markdown heading.
    """
    text = section_md.strip()

    # Remove a literal leaked placeholder like "<Section Title>" (any casing/spacing)
    text = re.sub(r"<\s*section\s*title\s*>", "", text, flags=re.IGNORECASE).strip()

    # Ensure the section actually starts with an H2 heading
    if not text.startswith("## "):
        first_line, _, rest = text.partition("\n")
        first_line = first_line.strip()
        if first_line:
            text = f"## {first_line}\n{rest}"
        else:
            text = f"## {fallback_title}\n\n{rest}"

    return text.strip()


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    if task.requires_research or task.requires_citations:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:8]
        ) or "(none)"
    else:
        evidence_text = "(not needed for this section)"

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    section_md = _clean_section_heading(section_md, task.title)

    return {"sections": [(task.id, section_md)]}

# -----------------------------
# 8) Reducer (merge + save)
# -----------------------------
def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def reducer_node(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("reducer_node called without plan.")

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}

# -----------------------------
# 9) Build main graph
# -----------------------------
def _timed(name: str, fn):
    """Wraps a node function to log wall-clock start/end/duration.
    Start/end timestamps let you see whether 'parallel' workers actually
    overlap in wall-clock time, or queue up sequentially on the Ollama side."""
    def wrapped(payload):
        t0 = time.perf_counter()
        started_at = time.strftime("%H:%M:%S")
        print(f"[timing] {name} START at {started_at}")
        try:
            result = fn(payload)
            return result
        finally:
            elapsed = time.perf_counter() - t0
            ended_at = time.strftime("%H:%M:%S")
            print(f"[timing] {name} END   at {ended_at}  ({elapsed:.1f}s)")
    return wrapped


g = StateGraph(State)
g.add_node("router", _timed("router", router_node))
g.add_node("research", _timed("research", research_node))
g.add_node("orchestrator", _timed("orchestrator", orchestrator_node))
g.add_node("worker", _timed("worker", worker_node))
g.add_node("reducer", _timed("reducer", reducer_node))

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app