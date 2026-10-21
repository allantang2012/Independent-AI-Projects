import re

from multi_agent import app

question = (
    "How does AutoGen enable communication "
    "and collaboration between multiple AI agents?"
)

result = app.invoke({
    "question": question,
    "evidence": "",
    "research": "",
    "analysis": "",
    "final_report": "",
    "next_agent": ""
})

report = result["final_report"]


print("\n--- CITATION HANDOFF DIAGNOSTICS ---")

for agent, field in [
    ("Researcher", "research"),
    ("Analyst", "analysis"),
    ("Writer", "final_report")
]:
    text = result.get(field, "")

    filename_found = "2308.08155v2.pdf" in text
    page_found = bool(
        re.search(r"\bp(?:age|\.)\s*\d+", text, re.I)
    )

    print(f"\n{agent}:")
    print(f"  Filename present: {filename_found}")
    print(f"  Page reference present: {page_found}")
    print(f"  Output length: {len(text)} characters")


# Check for expected components
checks = {
    "Report is not empty": bool(report.strip()),

    "Mentions AutoGen": (
        "autogen" in report.lower()
    ),

    "Includes PDF source filename": (
        "2308.08155v2.pdf" in report
    ),

    "Includes page references": bool(
        re.search(r"\bp(?:age|\.)\s*\d+", report, re.I)
    ),

    "Discusses agent communication": any(
        term in report.lower()
        for term in [
            "communication",
            "conversation",
            "messages"
        ]
    )
}

print("\n--- REPORT QUALITY CHECKS ---")

passed = 0

for name, success in checks.items():
    status = "PASS" if success else "FAIL"
    print(f"{status}: {name}")

    if success:
        passed += 1

print(f"\nChecks passed: {passed}/{len(checks)}")

with open("final_research_report.txt", "w") as file:
    file.write(report)

print("\nReport saved to final_research_report.txt")
