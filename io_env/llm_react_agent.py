#!/usr/bin/env python3
"""
LLM-Driven ReAct Agent for IO-Aware Operator Optimization.

Uses a real LLM (via Papyrus/Azure/Copilot) to drive the
Thought → Action → Observation loop over the IO Environment.

Usage:
    # Default: Papyrus GPT-5.2
    python -m io_env.llm_react_agent cross_entropy

    # Specify model
    python -m io_env.llm_react_agent cross_entropy --model gpt-5.2-chat_2025-12-11

    # All tasks
    python -m io_env.llm_react_agent all
"""

from __future__ import annotations
import json, re, time, os, sys, requests
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_env.react_agent import IOEnvironment, TOOLS


# ============================================================================
# Lightweight LLM client (self-contained, no external dependency on Agent_lhb)
# ============================================================================

class LLMClient:
    """Minimal LLM client supporting Papyrus and OpenAI-compatible APIs."""

    def __init__(self, model: str = "gpt-5.2", provider: str = "papyrus"):
        self.model = model
        self.provider = provider
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.num_calls = 0

        if provider == "papyrus":
            self._init_papyrus()
        elif provider == "copilot":
            self._init_copilot()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _init_papyrus(self):
        from azure.identity import AzureCliCredential
        self.endpoint = "https://westus2.papyrus.binginternal.com/chat/completions"
        self.verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"
        self.credential = AzureCliCredential()
        self._refresh_token()

    def _refresh_token(self):
        access_token = self.credential.get_token(self.verify_scope).token
        self.headers = {
            "Authorization": "Bearer " + access_token,
            "Content-Type": "application/json",
            "papyrus-model-name": self.model,
            "papyrus-quota-id": "PapyrusCustomer",
            "papyrus-timeout-ms": "120000",
        }

    def _init_copilot(self):
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:15432/v1", api_key="dummy")

    def chat(self, messages: list[dict]) -> str:
        if self.provider == "papyrus":
            return self._chat_papyrus(messages)
        elif self.provider == "copilot":
            return self._chat_copilot(messages)

    def _chat_papyrus(self, messages: list[dict]) -> str:
        json_body = {"messages": messages}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=json_body)
                response.raise_for_status()
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content']
                usage = resp_json.get('usage', {})
                self.total_prompt_tokens += usage.get('prompt_tokens', 0)
                self.total_completion_tokens += usage.get('completion_tokens', 0)
                self.num_calls += 1
                return content
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status == 401 and attempt < max_retries - 1:
                    print(f"  [LLM] Token expired, refreshing...")
                    self._refresh_token()
                elif status in (429, 408, 400) and attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"  [LLM] Rate limited ({status}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def _chat_copilot(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        content = response.choices[0].message.content
        usage = response.usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.num_calls += 1
        return content


# ============================================================================
# Action parser: extract tool call from LLM response
# ============================================================================

def parse_action(response: str) -> tuple[str, str, dict]:
    """
    Parse LLM response to extract Thought and Action.

    Expected format:
        Thought: <reasoning>
        Action: {"tool": "<name>", "args": {...}}

    Returns: (thought, tool_name, args_dict)
    """
    # Extract thought
    thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\Z)', response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else response.strip()

    # Extract action JSON
    action_match = re.search(r'Action:\s*(\{.*\})', response, re.DOTALL)
    if not action_match:
        # Try to find any JSON object in the response
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
        if json_match:
            action_match = json_match

    if action_match:
        try:
            action_str = action_match.group(1) if hasattr(action_match, 'group') else action_match.group(0)
            action = json.loads(action_str)
            return thought, action.get("tool", ""), action.get("args", {})
        except json.JSONDecodeError:
            pass

    # Fallback: try to detect intent from text
    lower = response.lower()
    if "done" in lower and ("complete" in lower or "finish" in lower):
        return thought, "done", {}

    return thought, "", {}


# ============================================================================
# LLM ReAct Loop
# ============================================================================

def run_llm_react_agent(task: str, model: str = "gpt-5.2",
                         provider: str = "papyrus",
                         max_steps: int = 8,
                         verbose: bool = True) -> IOEnvironment:
    """
    Run the full LLM-driven ReAct loop.

    1. Initialize environment with task
    2. Generate system prompt with tool descriptions
    3. Loop: LLM generates Thought+Action → Environment returns Observation
    4. Stop when LLM calls 'done' or max_steps reached
    """
    env = IOEnvironment()
    llm = LLMClient(model=model, provider=provider)

    if verbose:
        print("\n" + "━" * 60)
        print(f"  LLM ReAct Agent: {model} optimizing '{task}'")
        print("━" * 60)

    # System prompt
    system_prompt = env.get_prompt()

    # Initialize conversation with first observation
    initial_obs = env.reset(task)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Optimize this operator. Here is the initial analysis:\n\n{initial_obs}\n\nBegin by examining the IO bottlenecks and proposing your first optimization."},
    ]

    if verbose:
        print(f"\n┌─ Initial Observation ─────────────────────")
        for line in initial_obs.split("\n"):
            print(f"│ {line}")
        print(f"└────────────────────────────────────────────")

    for step in range(max_steps):
        # Get LLM response
        if verbose:
            print(f"\n  🤖 Calling {model}...", end="", flush=True)

        response = llm.chat(messages)

        if verbose:
            print(f" done ({llm.num_calls} calls, {llm.total_prompt_tokens + llm.total_completion_tokens} tokens)")

        # Parse thought + action
        thought, tool, args = parse_action(response)

        if verbose:
            print(f"\n┌─ Step {step} ─────────────────────────────────")
            print(f"│ Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
            print(f"│ Action:  {tool}({json.dumps(args, ensure_ascii=False)})")

        if not tool:
            # LLM didn't produce a valid action — add error and retry
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                "I couldn't parse your action. Please respond in this exact format:\n"
                "Thought: <your reasoning>\n"
                'Action: {"tool": "<tool_name>", "args": {<args>}}\n\n'
                f"Available tools: {list(TOOLS.keys())}"})
            if verbose:
                print(f"│ ⚠ Could not parse action, asking LLM to retry")
                print(f"└────────────────────────────────────────────")
            continue

        # Execute action
        obs, reward, done = env.step(thought, tool, args)

        if verbose:
            print(f"├─ Observation ──────────────────────────────")
            for line in obs.split("\n")[:15]:
                print(f"│ {line}")
            if len(obs.split("\n")) > 15:
                print(f"│ ... ({len(obs.split(chr(10))) - 15} more lines)")
            print(f"│ Reward: {reward:+.2f}")
            print(f"└────────────────────────────────────────────")

        # Add to conversation
        messages.append({"role": "assistant", "content": response})

        # Contextual nudge based on what just happened
        if tool == "verify" and "ALL PASSED" in obs:
            nudge = "Verification passed. Now run 'benchmark' to measure actual GPU speedup, then reflect on the results."
        elif tool == "benchmark":
            nudge = "You've seen the actual GPU benchmark. Reflect on why the actual speedup differs from the roofline prediction. Then call 'done' with your analysis."
        else:
            nudge = "Continue optimizing or call 'verify' if you think optimization is complete."

        messages.append({"role": "user", "content": f"Observation:\n{obs}\nReward: {reward:+.2f}\n\n{nudge}"})

        if done:
            break

    # Print final stats
    if verbose:
        print(f"\n{'='*60}")
        print(f"  LLM Stats: {llm.num_calls} calls, "
              f"{llm.total_prompt_tokens} prompt + {llm.total_completion_tokens} completion tokens")
        print(f"  Final IO: {env.baseline_report.total_io/1e6:.1f} MB → {env.current_report.total_io/1e6:.1f} MB "
              f"({(1-env.current_report.total_io/env.baseline_report.total_io)*100:.1f}% reduction)")
        print(f"{'='*60}")

    return env


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-Driven ReAct Agent for IO Optimization")
    parser.add_argument("task", nargs="?", default="cross_entropy",
                       help="Task: gmm_estep, kmeans, softmax, cross_entropy, all")
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument("--provider", default="papyrus", choices=["papyrus", "copilot"],
                       help="LLM provider")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["cross_entropy", "gmm_estep", "kmeans", "softmax"]
    else:
        tasks = [args.task]

    for task in tasks:
        run_llm_react_agent(task, model=args.model, provider=args.provider,
                           max_steps=args.max_steps, verbose=True)
