# from __future__ import annotations
# import operator
# from pathlib import Path
# from typing import TypedDict, List, Annotated
# from pydantic import BaseModel, Field
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Send
# from langchain_ollama import ChatOllama
# from langchain_core.messages import SystemMessage, HumanMessage

# class Task(BaseModel):
#     id: int
#     title: str
#     brief: str = Field(..., description='What to cover')

# class Plan(BaseModel):
#     blog_title: str
#     tasks: List[Task]

# class State(TypedDict):
#     topic: str
#     plan: Plan
#     sections: Annotated[List[str], operator.add]
#     final: str

# llm = ChatOllama(model='gemma3:4b')

# def orchestrator(state: State) -> dict:
#     plan = llm.with_structured_output(Plan).invoke([
#         SystemMessage(content='Create a blog plan with 5-7 sections on the following topic.'),
#         HumanMessage(content=f'Topic: {state["topic"]}')
#     ])
#     print('PLAN:', plan)
#     print('PLAN TYPE:', type(plan))
#     print('PLAN JSON:', plan.json() if hasattr(plan, 'json') else repr(plan))
#     return {'plan': plan}


# def fanout(state: State):
#     print('FANOUT: tasks', len(state['plan'].tasks), [t.title for t in state['plan'].tasks])
#     return [Send('worker', {'task': task, 'topic': state['topic'], 'plan': state['plan']}) for task in state['plan'].tasks]


# def worker(payload: dict) -> dict:
#     task = payload['task']
#     topic = payload['topic']
#     plan = payload['plan']
#     blog_title = plan.blog_title
#     section_md = llm.invoke([
#         SystemMessage(content='Write one clean Markdown section.'),
#         HumanMessage(content=(
#             f'Blog: {blog_title}\n'
#             f'Topic: {topic}\n\n'
#             f'Section: {task.title}\n'
#             f'Brief: {task.brief}\n\n'
#             'Return only the section content in Markdown.'
#         )),
#     ])
#     print('SECTION RAW:', repr(section_md))
#     content = section_md.content.strip() if hasattr(section_md, 'content') else str(section_md).strip()
#     print('SECTION LEN:', len(content))
#     print('SECTION PREVIEW:', repr(content[:200]))
#     return {'sections': [content]}


# def reducer(state: State) -> dict:
#     title = state['plan'].blog_title
#     body = '\n\n'.join(state['sections']).strip()
#     final_md = f'# {title}\n\n{body}\n'
#     print('FINAL LEN:', len(final_md))
#     print('FINAL PREVIEW:', repr(final_md[:200]))
#     Path(title.lower().replace(' ', '_') + '.md').write_text(final_md, encoding='utf-8')
#     return {'final': final_md}


# g = StateGraph(State)
# g.add_node('orchestrator', orchestrator)
# g.add_node('worker', worker)
# g.add_node('reducer', reducer)
# g.add_edge(START, 'orchestrator')
# g.add_conditional_edges('orchestrator', fanout, ['worker'])
# g.add_edge('worker', 'reducer')
# g.add_edge('reducer', END)
# app = g.compile()

# try:
#     out = app.invoke({'topic': 'Write a blog on Self Attention', 'sections': []})
#     print('OUT:', out)
# except Exception as e:
#     print('ERROR:', type(e).__name__, e)
import langgraph
print(langgraph.__version__)