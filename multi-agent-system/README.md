# Multi-Agent AI Research System

A tool-augmented multi-agent research workflow built with Python,
LangGraph, LangChain, and OpenAI models.

The system coordinates specialized agents to retrieve evidence
from PDF research papers, analyze findings, and generate
structured reports with source citations.

## Architecture

```text
                     User Question
                           |
                           v
                   +---------------+
                   |  Coordinator  |
                   +---------------+
                     /           \
            Research needed?    Research exists?
                   |                 |
                   v                 |
             +------------+          |
             | Researcher |          |
             +------------+          |
                |       |            |
                v       v            |
          Knowledge    PDF           |
            Base      Search         |
                |       |            |
                +---+---+            |
                    |                |
                    v                v
                   +------------------+
                   |     Analyst      |
                   +------------------+
                            |
                            v
                   +------------------+
                   |      Writer      |
                   +------------------+
                            |
                            v
                    Research Report
```

## Features

- Four specialized components: Coordinator, Researcher,
  Analyst, and Writer
- Conditional routing using LangGraph shared state
- LLM-directed tool selection using LangChain
- PDF document retrieval using TF-IDF and cosine similarity
- Source attribution with filenames and page numbers
- Tool-call limiting to control unnecessary API usage
- Execution metrics for model calls, tool calls, tokens,
  and latency
- Automated routing and structural report-quality tests

## Technology Stack

- Python
- LangGraph
- LangChain
- OpenAI API
- scikit-learn
- pypdf

## Evaluation Results

### Routing Tests

3/3 routing tests passed after switching from
LLM-based routing to deterministic routing.

### Report Quality Checks

5/5 structural checks passed, including:
- Report generated
- Correct research topic mentioned
- PDF source filename included
- Page references included
- Agent communication discussed

These checks do not establish complete factual accuracy.

### Performance

| Metric | Before Tool Limit | After Tool Limit |
|--------|-------------------|------------------|
| Execution time | 53.161 s | 36.318 s |
| Model calls | 8 | 6 |
| Tool calls | 5 | 3 |
| Total tokens | 17,045 | 8,214 |

Observed token reduction: 51.8%.

Results are from individual runs and should not be interpreted
as statistically reliable average performance.

## Getting Started

1. Clone the repository.
2. Navigate to `multi-agent-system`.
3. Create and activate a Python virtual environment.
4. Install the required dependencies.
5. Create a `.env` file with `OPENAI_API_KEY`.
6. Place text-based research PDFs in the `documents/` folder.
7. Run `python multi_agent.py`.

Example installation:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install langgraph langchain langchain-openai \
    python-dotenv pypdf scikit-learn
```

## Evaluation

Run:

```bash
python evaluate_agents.py
python test_routing.py
python test_report_quality.py
```

These scripts make OpenAI API calls and may incur charges.

## Limitations

- Current retrieval uses keyword-based TF-IDF, not embeddings.
- PDFs are indexed by page rather than smaller text chunks.
- Search indices are rebuilt for each retrieval request.
- Citation presence does not guarantee citation correctness.
- Evaluations use a small number of test scenarios.
- The system is a prototype, not a production-ready platform.

## Future Improvements

- Semantic retrieval with embeddings and vector search
- Retrieval-quality benchmarks
- More comprehensive source-grounding evaluation
- Dynamic supervisor-based delegation
- Per-agent latency and cost analysis
```

## Step 2: Make one important adjustment

Because your project has an existing root README, this new file will document **only the multi-agent project**.

Your repository structure will look approximately like:

```text
Independent-AI-Projects/
├── README.md
├── decoder-transformer/
├── gpt2-fine-tuning/
└── multi-agent-system/
    ├── README.md
    ├── multi_agent.py
    ├── research_tools.py
    ├── document_retriever.py
    ├── evaluate_agents.py
    ├── test_routing.py
    ├── test_report_quality.py
    └── documents/
```

One thing to check: if your root `.gitignore` excludes all `README.md` files or Markdown files, the new README might not be staged. We can verify that shortly.

## Step 3: Save and preview

After pasting the README, press **Command + S**.

Then run:

```bash
cd ~/Independent-AI-Projects

git status
```

You should see something like:

```text
Untracked files:
    multi-agent-system/README.md
```

If so, run:

```bash
git add multi-agent-system/README.md
git commit -m "Document multi-agent research system architecture and benchmarks"
git push
```

## Step 4: View your project on GitHub

After pushing, open:

<Link url="https://github.com/allantang2012/Independent-AI-Projects/tree/main/multi-agent-system" title="Your multi-agent project on GitHub"/>

GitHub should automatically display the new README beneath your files.

**For now, complete Steps 1–3 and tell me whether `git push` succeeds.** After that, we'll do the final repository check and write two strong, quantitatively grounded resume bullets for this project.


## MCP + A2A Integration

This project also demonstrates communication across two
AI-agent protocols:

- Model Context Protocol (MCP) for PDF retrieval.
- Agent2Agent Protocol (A2A) for communication with an
  independently running Analyst Agent.

### Architecture

Researcher
    |
    v
MCP PDF Retrieval Server
    |
    v
LLM Research Summarization
    |
    v
A2A HTTP / JSON-RPC
    |
    v
Independent Analyst Agent
    |
    v
Evidence-Based Analysis

### Run the Integrated Workflow

Install dependencies:

    pip install -r requirements.txt

Create a .env file containing OPENAI_API_KEY and add
a text-based PDF to the documents/ directory.

Terminal 1 — Start the A2A Analyst server:

    python a2a_agents/analyst_server.py

Terminal 2 — Run the integrated workflow:

    python mcp_a2a_workflow.py

The workflow starts the MCP server automatically,
retrieves PDF evidence, summarizes the findings,
and sends them to the independent Analyst over A2A.

### Verified Capabilities

- MCP tool discovery and execution
- PDF retrieval with page references
- A2A Agent Card discovery
- HTTP/JSON-RPC communication
- Independent LLM-powered Analyst
- Source-aware research handoffs

### Limitations

The combined workflow currently uses a predefined
MCP search followed by an A2A handoff. It does not
implement autonomous multi-agent task delegation.
