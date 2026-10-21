
from multi_agent import app

test_cases = [
    {
        "name": "Research missing",
        "question": "What are the benefits of AI agents?",
        "research": "",
        "expected_route": "researcher"
    },
    {
        "name": "Research already provided",
        "question": "What are the benefits of AI agents?",
        "research": (
            "AI agents automate multi-step tasks "
            "and reduce repetitive work."
        ),
        "expected_route": "analyst"
    },
    {
        "name": "Reliability research",
        "question": "How can AI agent reliability be evaluated?",
        "research": "",
        "expected_route": "researcher"
    }
]

passed = 0

for case in test_cases:
    print(f"\n--- Testing: {case['name']} ---")

    result = app.invoke({
        "question": case["question"],
        "evidence": "",
        "research": case["research"],
        "analysis": "",
        "final_report": "",
        "next_agent": ""
    })

    actual_route = result["next_agent"]
    route_correct = actual_route == case["expected_route"]
    report_generated = bool(result["final_report"].strip())

    if route_correct and report_generated:
        passed += 1
        print("PASS")
    else:
        print("FAIL")

    print("Expected route:", case["expected_route"])
    print("Actual route:", actual_route)
    print("Report generated:", report_generated)

print(f"\nRouting tests passed: {passed}/{len(test_cases)}")
