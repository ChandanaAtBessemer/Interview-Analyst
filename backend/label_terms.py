# label_terms.py  – works with openai-python ≥1.0
import json, textwrap
from dotenv import load_dotenv

load_dotenv()                       # pulls OPENAI_API_KEY from .env

from openai import OpenAI           # AFTER load_dotenv so key is in env
client = OpenAI()

_SYSTEM = """
You are a taxonomist for interview-analysis.
Ignore generic helper verbs (be, do, have, make, take, get, go, say, want).
Buckets: issues, actions, solutions, benefits, emotions, qualities,
comparisons, references, time, frequency.
Return JSON on each line: {"term": "...", "categories": [...] }
""".strip()

_FEW = [
    {"role": "user", "content": "downtime"},
    {"role": "assistant", "content": '{"term":"downtime","categories":["issues","time"]}'},
    {"role": "user", "content": "upgrade"},
    {"role": "assistant", "content": '{"term":"upgrade","categories":["actions","solutions"]}'},
]

def label_batch(terms: list[str]) -> dict[str, list[str]]:
    messages = [{"role": "system", "content": _SYSTEM}] + _FEW
    for t in terms:
        messages.append({"role": "user", "content": t})

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=messages
    )

    lines = resp.choices[0].message.content.strip().split("\n")
    return { (d := json.loads(l))["term"] : d["categories"] for l in lines }
