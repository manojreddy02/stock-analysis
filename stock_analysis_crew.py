"""
Stock Market Fundamental Analysis Agent System
Built with CrewAI v1.11+ | Azure OpenAI

Prerequisites:
  1. Python >= 3.10, < 3.14
  2. Install packages:
       pip install crewai 'crewai[tools]' python-dotenv
  3. Create a .env file in the same directory (see .env.example below)
  4. Get a free Serper API key at: https://serper.dev/

.env.example:
  # ── Azure OpenAI ──
  AZURE_API_KEY=your-azure-openai-api-key
  AZURE_API_BASE=https://your-resource-name.openai.azure.com/
  AZURE_API_VERSION=2024-08-01-preview
  AZURE_DEPLOYMENT_NAME=gpt-4o

  # ── Serper (Web Search) ──
  SERPER_API_KEY=your-serper-api-key

Usage:
  python stock_analysis_crew.py                  # Analyzes default ticker (AAPL)
  python stock_analysis_crew.py TSLA             # Analyzes Tesla
  python stock_analysis_crew.py MSFT --verbose   # Verbose mode
"""

import os
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


# ─────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────

load_dotenv()

# Validate required env vars
required_vars = {
    "AZURE_API_KEY": os.getenv("AZURE_API_KEY"),
    "AZURE_API_BASE": os.getenv("AZURE_API_BASE"),
    "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
    "AZURE_DEPLOYMENT_NAME": os.getenv("AZURE_DEPLOYMENT_NAME"),
    "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
}

missing = [k for k, v in required_vars.items() if not v]
if missing:
    print(f"ERROR: Missing required environment variables in .env file:")
    for var in missing:
        print(f"  - {var}")
    print("\nSee the .env.example section in this script's docstring.")
    sys.exit(1)


# ─────────────────────────────────────────────
# LLM CONFIGURATION (Azure OpenAI)
# ─────────────────────────────────────────────

# Some Azure model deployments reject the 'stop' parameter. Patch the native
# Azure provider to skip sending stop words (they are still applied locally).
try:
    from crewai.llms.providers.azure.completion import AzureCompletion
    AzureCompletion.supports_stop_words = lambda self: False
except ImportError:
    pass

azure_llm = LLM(
    model=f"azure/{required_vars['AZURE_DEPLOYMENT_NAME']}",
    api_key=required_vars["AZURE_API_KEY"],
    base_url=required_vars["AZURE_API_BASE"],
    api_version=required_vars["AZURE_API_VERSION"],
    temperature=0.3,
)


# ─────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────

ticker = sys.argv[1].upper() if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "AAPL"
verbose = "--verbose" in sys.argv


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

search_tool = SerperDevTool(n_results=10)
scrape_tool = ScrapeWebsiteTool()


# ─────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────

financial_researcher = Agent(
    role="Senior Financial Researcher",
    goal=(
        f"Gather comprehensive financial data and recent news about {ticker}. "
        "Find revenue, earnings, balance sheet data, recent SEC filings, "
        "analyst ratings, and any major news that could impact the stock."
    ),
    backstory=(
        "You are a seasoned financial researcher with 15+ years of experience "
        "at top investment banks. You are meticulous about finding accurate, "
        "up-to-date financial data from reliable sources like SEC filings, "
        "earnings reports, and reputable financial news outlets. You always "
        "cross-reference data from multiple sources."
    ),
    tools=[search_tool, scrape_tool],
    llm=azure_llm,
    verbose=verbose,
    memory=True,
    allow_delegation=False,
    max_iter=15,
)

financial_analyst = Agent(
    role="Senior Fundamental Analyst",
    goal=(
        f"Perform deep fundamental analysis of {ticker} using the research data. "
        "Calculate and interpret key financial ratios, assess the company's "
        "competitive position, evaluate management quality, and identify "
        "strengths, weaknesses, opportunities, and threats."
    ),
    backstory=(
        "You are a CFA charterholder with deep expertise in equity valuation "
        "and fundamental analysis. You've evaluated hundreds of public companies "
        "across sectors. You focus on financial health, growth prospects, "
        "profitability metrics, and competitive moats. You think critically "
        "and always consider both bull and bear cases."
    ),
    tools=[search_tool],
    llm=azure_llm,
    verbose=verbose,
    memory=True,
    allow_delegation=False,
    max_iter=10,
)

report_writer = Agent(
    role="Investment Report Writer",
    goal=(
        f"Compile a professional, well-structured fundamental analysis report "
        f"for {ticker} that is clear, actionable, and suitable for investors."
    ),
    backstory=(
        "You are an experienced financial writer who has authored reports for "
        "major investment firms. You excel at translating complex financial "
        "data into clear, readable analysis. Your reports are known for being "
        "thorough yet concise, always including both quantitative metrics and "
        "qualitative insights. You never provide direct buy/sell recommendations "
        "but present the data so investors can make informed decisions."
    ),
    tools=[],
    llm=azure_llm,
    verbose=verbose,
    memory=True,
    allow_delegation=False,
    max_iter=5,
)


# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────

research_task = Task(
    description=(
        f"Research the stock {ticker} thoroughly. Your research MUST cover:\n\n"
        f"1. **Company Overview**: What does {ticker} do? What sector/industry?\n"
        f"2. **Recent Financial Performance**: Latest quarterly and annual revenue, "
        f"   net income, EPS, and revenue growth trends (last 3-4 quarters).\n"
        f"3. **Balance Sheet Health**: Total debt, cash position, debt-to-equity ratio.\n"
        f"4. **Recent News & Events**: Any major news in the last 30 days — "
        f"   earnings surprises, product launches, management changes, lawsuits, "
        f"   regulatory actions, partnerships, M&A activity.\n"
        f"5. **Analyst Consensus**: Current analyst ratings, average price target, "
        f"   and any recent upgrades/downgrades.\n"
        f"6. **Competitive Landscape**: Key competitors and how {ticker} compares.\n\n"
        f"Use web search to find the most current data. Cross-reference at least "
        f"2-3 sources for key financial figures. Cite your sources."
    ),
    expected_output=(
        "A detailed research dossier organized by the 6 categories above, "
        "with specific numbers, dates, and source references. "
        "The data should be current and accurate."
    ),
    agent=financial_researcher,
)

analysis_task = Task(
    description=(
        f"Using the research data gathered about {ticker}, perform a comprehensive "
        f"fundamental analysis. Your analysis MUST include:\n\n"
        f"1. **Key Financial Ratios** (calculate or find current values):\n"
        f"   - P/E Ratio (trailing and forward)\n"
        f"   - P/S Ratio\n"
        f"   - P/B Ratio\n"
        f"   - Debt-to-Equity Ratio\n"
        f"   - Current Ratio\n"
        f"   - Return on Equity (ROE)\n"
        f"   - Profit Margins (gross, operating, net)\n"
        f"   - Free Cash Flow Yield\n\n"
        f"2. **Growth Analysis**: Revenue and earnings growth trajectory, "
        f"   compare to industry averages.\n\n"
        f"3. **Valuation Assessment**: Is the stock overvalued, fairly valued, "
        f"   or undervalued based on the ratios? Compare to sector peers.\n\n"
        f"4. **SWOT Analysis**: Strengths, Weaknesses, Opportunities, Threats.\n\n"
        f"5. **Risk Factors**: What are the key risks investors should consider?\n\n"
        f"6. **Bull vs Bear Case**: Present both perspectives with supporting data.\n\n"
        f"If you need to look up any missing data points, use web search."
    ),
    expected_output=(
        "A structured fundamental analysis covering all 6 sections above, "
        "with specific numbers and comparisons. Each conclusion should be "
        "supported by data points from the research."
    ),
    agent=financial_analyst,
    context=[research_task],
)

report_task = Task(
    description=(
        f"Write a professional Fundamental Analysis Report for {ticker}. "
        f"Use the research and analysis provided to create a polished report.\n\n"
        f"The report MUST follow this structure:\n\n"
        f"# Fundamental Analysis Report: {ticker}\n"
        f"**Date**: [Today's date]\n\n"
        f"## Executive Summary\n"
        f"A 3-4 sentence overview of the key findings.\n\n"
        f"## Company Overview\n"
        f"Brief description of the business, sector, and market position.\n\n"
        f"## Financial Performance\n"
        f"Revenue, earnings, and growth trends with specific numbers.\n\n"
        f"## Key Financial Ratios\n"
        f"Present ratios in a clear format with peer comparisons.\n\n"
        f"## Valuation Analysis\n"
        f"Is it overvalued/undervalued? Support with data.\n\n"
        f"## SWOT Analysis\n"
        f"Strengths, Weaknesses, Opportunities, Threats.\n\n"
        f"## Risk Factors\n"
        f"Key risks investors should monitor.\n\n"
        f"## Bull vs Bear Case\n"
        f"Both perspectives with supporting evidence.\n\n"
        f"## Conclusion\n"
        f"Summary of findings. Do NOT give a buy/sell recommendation — "
        f"instead, highlight the key factors investors should weigh.\n\n"
        f"**Important**: Include a disclaimer that this is AI-generated analysis "
        f"and not financial advice."
    ),
    expected_output=(
        "A complete, professionally formatted markdown report following "
        "the structure above. The report should be 1500-2500 words, "
        "data-driven, and balanced in its assessment."
    ),
    agent=report_writer,
    context=[research_task, analysis_task],
    output_file=f"{ticker}_fundamental_analysis.md",
)


# ─────────────────────────────────────────────
# CREW ASSEMBLY & EXECUTION
# ─────────────────────────────────────────────

crew = Crew(
    agents=[financial_researcher, financial_analyst, report_writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,
    verbose=verbose,
    memory=True,
    cache=True,
    max_rpm=100,
)


def main():
    print(f"\n{'='*60}")
    print(f"  Stock Fundamental Analysis Agent System")
    print(f"  Analyzing: {ticker}")
    print(f"  LLM: Azure OpenAI ({required_vars['AZURE_DEPLOYMENT_NAME']})")
    print(f"{'='*60}\n")

    result = crew.kickoff(inputs={"company": ticker, "ticker": ticker})

    print(f"\n{'='*60}")
    print(f"  Analysis Complete!")
    print(f"  Report saved to: {ticker}_fundamental_analysis.md")
    print(f"{'='*60}\n")

    # Print token usage summary if available
    if hasattr(result, "token_usage"):
        usage = result.token_usage
        print(f"Token Usage Summary:")
        print(f"  Total tokens:  {usage.total_tokens:,}")
        print(f"  Prompt tokens: {usage.prompt_tokens:,}")
        print(f"  Output tokens: {usage.completion_tokens:,}")
        print()

    print(result.raw)


if __name__ == "__main__":
    main()
