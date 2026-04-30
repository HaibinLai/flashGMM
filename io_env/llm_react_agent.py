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
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=json_body, timeout=120)
                response.raise_for_status()
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content']
                usage = resp_json.get('usage', {})
                self.total_prompt_tokens += usage.get('prompt_tokens', 0)
                self.total_completion_tokens += usage.get('completion_tokens', 0)
                self.num_calls += 1
                return content
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                status = getattr(getattr(e, 'response', None), 'status_code', None)
                if status == 401 and attempt < max_retries - 1:
                    print(f"  [LLM] Token expired, refreshing...")
                    time.sleep(5)
                    self._refresh_token()
                elif attempt < max_retries - 1:
                    wait = 10
                    print(f"  [LLM] Error ({status or type(e).__name__}), retry {attempt+1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                    if status == 401:
                        self._refresh_token()
                else:
                    print(f"  [LLM] Max retries ({max_retries}) reached. Last error: {e}")
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
        {"role": "user", "content": f"Optimize this operator. Here is the initial analysis:\n\n{initial_obs}\n\nFirst, run 'pre_check' to verify IO optimization is appropriate, then proceed with optimization."},
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
        if tool == "analyze":
            nudge = ("Analysis complete. Now run 'pre_check' to verify whether IO optimization "
                     "is appropriate for this operator (checks memory-bound, L2 cache, GEMM structure).")
        elif tool == "pre_check":
            if "CAUTION" in obs:
                nudge = ("Pre-check raised CAUTION. Read the warnings carefully. "
                         "IO optimization may not yield speedup for this operator. "
                         "Consider the recommendations before proceeding.")
            else:
                nudge = ("Pre-check passed. Proceed with IO optimization. "
                         "Apply fuse_and_online or fuse_ops to eliminate materialized intermediates.")
        elif tool == "verify" and "ALL PASSED" in obs:
            nudge = ("Verification passed! Now you MUST generate an actual Triton GPU kernel. "
                     "Call 'generate_kernel' to create the kernel, then 'compile_and_test' to verify it, "
                     "then 'benchmark_kernel' to measure real speedup.")
        elif tool == "generate_kernel":
            nudge = "Kernel generated. Now call 'compile_and_test' to compile and verify correctness."
        elif tool == "compile_and_test" and "PASS" in obs:
            nudge = "Kernel compiled and passed correctness test! Now call 'benchmark_kernel' to measure actual GPU speedup."
        elif tool == "compile_and_test" and "FAIL" in obs:
            nudge = ("Kernel correctness test FAILED. Analyze the error and write a fixed kernel "
                     "using generate_kernel with custom_code=<your corrected Triton code>.")
        elif tool == "benchmark_kernel":
            # Check if speedup < 1.0
            is_slow = "slower" in obs.lower() or "0." in obs.split("Speedup:")[-1].split("×")[0] if "Speedup:" in obs else False
            is_regression = "REGRESSION" in obs
            # Extract speedup value
            speedup_val = 1.0
            for bline in obs.split("\n"):
                if "Speedup:" in bline:
                    try: speedup_val = float(bline.split(":")[-1].strip().rstrip("×"))
                    except ValueError: pass

            if is_regression:
                nudge = ("⚠ Your new kernel is SLOWER than the previous best! "
                         "The environment auto-rolled back to the best kernel. "
                         "Call 'ncu_profile' on the best kernel to understand its limits. "
                         "Then try a FUNDAMENTALLY DIFFERENT approach, not small tweaks. "
                         "If you've already tried 2+ approaches, call 'done' — the best kernel is active.")
            elif is_slow:
                nudge = ("KERNEL IS SLOWER THAN BASELINE. Use data-driven diagnosis: "
                         "Step 1: Call 'ncu_profile' to get runtime bandwidth/compute utilization. "
                         "Step 2: Call 'library_ceiling' to see how far from optimal. "
                         "Step 3: Based on diagnosis, either rewrite kernel or accept result. "
                         "DO NOT guess — let the profiling data guide your fix.")
            elif speedup_val < 2.0:
                nudge = (f"Speedup is {speedup_val:.2f}× — decent but there may be room for improvement. "
                         "Call 'ncu_profile' to check bandwidth/compute utilization. "
                         "If bandwidth > 70%, you're near memory-bound limit — try 'autotune_kernel'. "
                         "If compute util is low, there may be a parallelism issue to fix. "
                         "Call 'library_ceiling' to see how far from optimal.")
            else:
                nudge = (f"Great speedup ({speedup_val:.2f}×)! But don't stop yet — "
                         "call 'ncu_profile' to check bandwidth utilization, "
                         "then 'library_ceiling' to see the ceiling, "
                         "then try 'autotune_kernel' to squeeze out more performance. "
                         "Only call 'done' after you've profiled AND tried autotune.")
        elif tool == "ncu_profile":
            if "under-utilized" in obs.lower():
                nudge = ("Kernel is UNDER-UTILIZED (low bandwidth AND compute). "
                         "This is a structural problem. Call 'compare_profile' to see what changed "
                         "vs baseline, then 'retrieve_pattern' to learn the correct pattern. "
                         "After fixing: generate_kernel → compile_and_test → benchmark_kernel → ncu_profile again.")
            elif "NOT using Tensor Core" in obs or "NO ✗" in obs:
                nudge = ("Kernel is NOT using Tensor Core. For distance/matmul patterns, "
                         "call 'retrieve_pattern' with 'gemm_online_reduce' to learn how to add tl.dot. "
                         "After fixing: generate_kernel → compile_and_test → benchmark_kernel → ncu_profile again.")
            elif "memory-bound" in obs.lower() and "good" in obs.lower():
                nudge = ("Kernel is memory-bound with good utilization — IO optimization is working! "
                         "Now call 'library_ceiling' to see the gap, then 'autotune_kernel' to try "
                         "different block sizes. Re-benchmark and re-profile after autotune.")
            elif "compute-bound" in obs.lower():
                if "Tensor Core" in obs and "YES" in obs:
                    nudge = ("Kernel is compute-bound WITH Tensor Core — good efficiency. "
                             "Call 'library_ceiling' to see if there's still a gap, "
                             "then try 'autotune_kernel'. Only 'done' after trying autotune.")
                else:
                    nudge = ("Kernel is compute-bound but WITHOUT Tensor Core. "
                             "Add tl.dot for Tensor Core usage. "
                             "After fixing: generate_kernel → compile_and_test → benchmark_kernel → ncu_profile again.")
        elif tool == "library_ceiling":
            if "Gap > 10" in obs or "fundamental" in obs.lower():
                nudge = ("Library is 10×+ faster — your kernel has fundamental issues. "
                         "If you haven't tried 'ncu_profile' yet, do that first to diagnose. "
                         "Otherwise, try a different algorithm (e.g., tl.dot for GEMM). "
                         "If you've already iterated 2+ times, accept and call 'done' with explanation.")
            elif "near optimal" in obs.lower() or "Within 1.5" in obs:
                nudge = ("Close to library ceiling! But before calling 'done', "
                         "try 'autotune_kernel' to sweep block sizes — you might gain another 10-30%. "
                         "After autotune, re-benchmark and re-profile to confirm final performance.")
            elif "Gap 1.5-3" in obs:
                nudge = ("Within 3× of library — try 'autotune_kernel' to close the gap, "
                         "then 'benchmark_kernel' and 'ncu_profile' to verify improvement.")
            else:
                nudge = ("There's room for improvement. Call 'ncu_profile' to identify "
                         "the bottleneck, fix it, then benchmark_kernel → ncu_profile again.")
        elif tool == "compare_profile":
            nudge = ("Review the side-by-side comparison above. Focus on the biggest difference "
                     "(bandwidth util, compute util, or TC usage) and fix that dimension first. "
                     "Then generate_kernel → compile_and_test → benchmark_kernel → ncu_profile.")
        elif tool == "autotune_kernel":
            nudge = ("Autotune complete! Now call 'benchmark_kernel' to measure the tuned kernel, "
                     "then 'ncu_profile' to verify utilization improved. "
                     "If this is your best result after profiling, you can call 'done'.")
        elif tool == "occupancy_analysis":
            if "LOW OCCUPANCY" in obs:
                nudge = ("Low occupancy detected. Adjust BLOCK_SIZE, num_warps, or reduce register pressure "
                         "in your kernel, then regenerate with 'generate_kernel'.")
            else:
                nudge = "Occupancy is acceptable. Focus on other optimization dimensions."
        elif tool == "benchmark":
            nudge = ("That was the Python-level benchmark. Now generate the REAL Triton kernel: "
                     "call 'generate_kernel', then 'compile_and_test', then 'benchmark_kernel'.")
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
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["cross_entropy", "gmm_estep", "kmeans", "softmax"]
    else:
        tasks = [args.task]

    for task in tasks:
        run_llm_react_agent(task, model=args.model, provider=args.provider,
                           max_steps=args.max_steps, verbose=True)
