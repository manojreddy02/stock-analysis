# Stock Market Fundamental Analysis — CrewAI Agent System

A multi-agent AI system built with [CrewAI](https://www.crewai.com/) and Azure OpenAI that performs comprehensive fundamental analysis on any publicly traded stock. Three specialized agents — researcher, analyst, and report writer — collaborate to deliver a professional investment analysis report using live web data.

## How It Works

Three specialized AI agents collaborate in sequence:

1. **Financial Researcher** — Searches the web for financial data, earnings reports, analyst ratings, recent news, and competitive landscape.
2. **Fundamental Analyst** — Interprets the research to calculate key ratios (P/E, P/B, ROE, margins, etc.), performs SWOT analysis, and builds bull/bear cases.
3. **Report Writer** — Compiles everything into a polished markdown report saved to a file.

---

## Prerequisites

- **Python** >= 3.10, < 3.14
- **Azure OpenAI** resource with a deployed model (e.g., `gpt-4o`)
- **Serper API key** for web search (free tier available)

---

## Installation

### 1. Clone the project

```bash
mkdir stock-analysis-crew && cd stock-analysis-crew
```

Place `stock_analysis_crew.py` and `.env.example` in this directory.

### 2. Install dependencies

```bash
pip install 'crewai[tools,azure-ai-inference]' python-dotenv
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your actual values:

```env
# ── Azure OpenAI ──
AZURE_API_KEY=your-azure-openai-api-key
AZURE_API_BASE=https://your-resource-name.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4o

# ── Serper (Web Search) ──
SERPER_API_KEY=your-serper-api-key
```

**Where to find your Azure values:**

| Variable | Where to find it |
|---|---|
| `AZURE_API_KEY` | Azure Portal → Your OpenAI resource → **Keys and Endpoint** → Key1 or Key2 |
| `AZURE_API_BASE` | Same page → **Endpoint** URL |
| `AZURE_API_VERSION` | Use `2024-08-01-preview` or check [Azure docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) for the latest |
| `AZURE_DEPLOYMENT_NAME` | Azure Portal → Your OpenAI resource → **Model deployments** → deployment name |

**Where to get a Serper key:**

Go to [serper.dev](https://serper.dev/), sign up, and copy your API key. The free tier gives you 2,500 searches.

---

## Usage

### Basic (defaults to AAPL)

```bash
python stock_analysis_crew.py
```

### Analyze a specific stock

```bash
python stock_analysis_crew.py TSLA
```

### Verbose mode (see agent reasoning step-by-step)

```bash
python stock_analysis_crew.py MSFT --verbose
```

### Output

The final report is saved as a markdown file in the current directory:

```
TSLA_fundamental_analysis.md
```

A token usage summary is also printed to the console after execution.

---

## Project Structure

```
stock-analysis-crew/
├── stock_analysis_crew.py    # Main script (agents, tasks, crew)
├── .env.example              # Environment variables template
├── .env                      # Your actual secrets (DO NOT commit)
└── README.md                 # This file
```

---

## Report Contents

The generated report includes:

- **Executive Summary** — Key findings at a glance
- **Company Overview** — Business description, sector, and market position
- **Financial Performance** — Revenue, earnings, and growth trends
- **Key Financial Ratios** — P/E, P/S, P/B, D/E, ROE, margins, FCF yield with peer comparisons
- **Valuation Analysis** — Overvalued / fairly valued / undervalued assessment
- **SWOT Analysis** — Strengths, weaknesses, opportunities, threats
- **Risk Factors** — Key risks to monitor
- **Bull vs Bear Case** — Both perspectives with supporting data
- **Conclusion** — Summary of factors investors should weigh

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ImportError: Azure AI Inference native provider not available` | Run `pip install 'crewai[azure-ai-inference]'` |
| `Missing required environment variables` | Check your `.env` file has all 5 variables filled in |
| `401 Unauthorized` from Azure | Verify `AZURE_API_KEY` and `AZURE_API_BASE` are correct |
| `DeploymentNotFound` | Make sure `AZURE_DEPLOYMENT_NAME` matches your Azure deployment exactly |
| Agent not using web search | Verify `SERPER_API_KEY` is valid — check credits at [serper.dev](https://serper.dev/) |
| Rate limited (429) | Azure is throttling — request a quota increase at [aka.ms/oai/quotaincrease](https://aka.ms/oai/quotaincrease) |

---

## Disclaimer

This tool generates AI-powered analysis for informational purposes only. It is **not financial advice**. Always do your own due diligence and consult a qualified financial advisor before making investment decisions.