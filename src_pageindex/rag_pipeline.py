"""
Reasoning-based RAG pipeline standardized on OpenAI (gpt-4o-mini).
"""

import json
from typing import List, Optional
from pydantic import BaseModel

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import pageindex.utils as pi_utils

from src_pageindex.config import OPENAI_API_KEY, OPENAI_MODEL_NAME

# ── OpenAI Client ──────────────────────────────────────────────────────────
_client = None

def _get_client():
    """Lazily initialize the OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment or config.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# ── OpenAI Interface ───────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=5, min=10, max=120),
    stop=stop_after_attempt(10),
    before_sleep=lambda rs: print(f"  ⏳ OpenAI Rate limited. Waiting... (attempt {rs.attempt_number}/10)")
)
def _call_openai_with_retry(messages: list, response_model=None, model_name: str = OPENAI_MODEL_NAME):
    """Call OpenAI with tenacity exponential backoff and optional JSON schema enforcement."""
    client = _get_client()

    if response_model:
        schema_dict = response_model.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2)
        instruction = f"\n\nIMPORTANT: You must respond ONLY with valid JSON exactly matching this schema:\n{schema_json}"
        
        # Ensure system instruction carries the schema
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()

        # Clean JSON markdown if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return response_model.model_validate_json(content)
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content

# ── Reasoning Based Retrieval ───────────────────────────────────────────────

class TreeSearchOutput(BaseModel):
    thinking: str
    node_list: List[str]

def run_reasoning_retrieval(query: str, tree: list) -> List[str]:
    """Uses LLM to evaluate document tree and pick relevant node IDs."""
    tree_without_text = pi_utils.remove_fields(tree.copy(), fields=['text'])
    
    search_prompt = f"""
You are given a question and a tree structure of a financial document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}
"""

    system_instruction = "You are an expert retrieval system mapping queries to node IDs."

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": search_prompt},
    ]

    print("    [Reasoning Retrieval] Running LLM tree search...")
    result = _call_openai_with_retry(messages, response_model=TreeSearchOutput)
    print(f"    [Reasoning Retrieval] Extracted {len(result.node_list)} relevant nodes.")
    return result.node_list


def run_pageindex_rag_query(query: str, tree: list, response_model=None):
    """Full RAG flow: Tree Search -> Text Extraction -> Final Answer Generation."""
    # 1. Tree Search (Reasoning Retrieval)
    relevant_node_ids = run_reasoning_retrieval(query, tree)
    
    # 2. Extract Text from the selected nodes
    node_map = pi_utils.create_node_mapping(tree)
    relevant_texts = []
    
    for nid in relevant_node_ids:
        node = node_map.get(nid)
        if node and "text" in node:
            txt = node["text"]
            if isinstance(txt, list):
                txt = "\n\n".join(txt)
            relevant_texts.append(f"--- Section: {node.get('title', 'Unknown')} ---\n{txt}")
            
    context_str = "\n\n".join(relevant_texts)
    
    # 3. Final Structured Generation
    final_prompt = f"""
Execute the following query based only on the following context retrieved from a proxy statement.

Query:
{query}

Context:
{context_str}
"""
    
    system_instruction = (
        "You are an expert at financial document analysis and providing structured responses."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": final_prompt},
    ]

    print("    [Generation] Generating final structured answer...")
    return _call_openai_with_retry(messages, response_model=response_model)
