import json
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


class AgentMetrics(BaseCallbackHandler):
    """Collect metrics from a LangGraph workflow."""

    def __init__(self):
        super().__init__()

        self.input_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.tool_names = []

    def on_llm_end(self, response, **kwargs):
        self.model_calls += 1

        # Read usage from the returned AI message
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)

                if message is None:
                    continue

                usage = getattr(message, "usage_metadata", None)

                if usage:
                    self.input_tokens += usage.get(
                        "input_tokens", 0
                    )
                    self.output_tokens += usage.get(
                        "output_tokens", 0
                    )

    def on_tool_start(
        self, serialized, input_str, **kwargs
    ):
        self.tool_calls += 1

        name = serialized.get("name", "unknown")
        self.tool_names.append(name)


def evaluate_workflow(app, test_input):
    metrics = AgentMetrics()

    start = time.perf_counter()

    result = app.invoke(
        test_input,
        config={"callbacks": [metrics]}
    )

    elapsed = time.perf_counter() - start

    report = {
        "execution_time_seconds": round(elapsed, 3),
        "model_calls": metrics.model_calls,
        "tool_calls": metrics.tool_calls,
        "tool_names": metrics.tool_names,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": (
            metrics.input_tokens + metrics.output_tokens
        ),
        "selected_agent": result.get("next_agent"),
        "report_generated": bool(
            result.get("final_report", "").strip()
        )
    }

    return result, report


if __name__ == "__main__":
    from multi_agent import app

    test_input = {
        "question": (
        "How does AutoGen enable communication "
        "and collaboration between multiple AI agents?"
        ),
        "evidence": "",
        "research": "",
        "analysis": "",
        "final_report": "",
        "next_agent": ""
    }

    result, metrics = evaluate_workflow(
        app,
        test_input
    )

    print("\n--- MULTI-AGENT EVALUATION ---")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save metrics to JSON
    Path("evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nMetrics saved to evaluation_metrics.json")
