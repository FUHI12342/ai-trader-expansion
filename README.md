# AI Trader Expansion

> AI-powered stock trading with Walk-Forward evaluation

Multi-strategy trading system for Japanese and US equities with rigorous out-of-sample validation, Monte Carlo stress testing, and a REST API for AI assistant integration.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-immutable-orange.svg)](docs/SPEC.md)

## Features

- **5 Trading Strategies** — MA Crossover, Dual Momentum (Antonacci), MACD+RSI, Bollinger+RSI+ADX, LightGBM Walk-Forward ML
- **Walk-Forward Analysis** — In-sample / out-of-sample split with t-test and bootstrap confidence intervals to prevent overfitting
- **Monte Carlo Simulation** — 1,000-shuffle robustness test with Sharpe / Sortino / Calmar metrics
- **Multi-Source Data** — yfinance (US + JP), J-Quants API (official JP equities), EDINET API v2 (disclosure documents), SQLite cache
- **Broker Adapters** — Paper trading (virtual orders + position management) and kabuSTATION (au Kabucom REST API)
- **SHANON REST API** — `/api/status`, `/api/positions`, `/api/performance`, `/api/backtest` endpoints for AI assistant integration
- **Production-Grade Design** — Immutable DataFrames, full type hints, environment-variable secrets, 80%+ test coverage

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourorg/ai-trader-expansion.git
cd ai-trader-expansion
pip install -r requirements.txt

# 2. Set API credentials (optional — yfinance works without keys)
export JQUANTS_REFRESH_TOKEN=your_token
export EDINET_API_KEY=your_key

# 3. Run a demo backtest
python scripts/demo_backtest.py

# 4. Start the REST API server
python -m src.api.server          # http://localhost:8765

# 5. Run all tests
pytest
```

## Architecture

```mermaid
graph TB
    subgraph DataSources["Data Sources"]
        YF[yfinance\nUS + JP equities]
        JQ[J-Quants API\nOfficial JP daily data]
        ED[EDINET API v2\nDisclosure documents]
    end

    subgraph DataLayer["Data Layer (src/data/)"]
        YFC[YFinanceClient]
        JQC[JQuantsClient\nauth + fetch]
        EDC[EdinetClient]
        DM[DataManager\nsource switching + SQLite cache]
    end

    subgraph StrategyLayer["Strategies (src/strategies/)"]
        BASE[BaseStrategy]
        MAC[MACrossover\nGolden/Dead cross]
        DMO[DualMomentum\nAntonacci method]
        MRS[MACD_RSI]
        BRA[BollingerRSI_ADX\nTriple filter]
        LGB[LGBMPredictor\nWalk-Forward ML]
    end

    subgraph EvalLayer["Evaluation (src/evaluation/)"]
        BT[Backtester\nEvent-driven\ncommission + slippage]
        WF[WalkForward\nIS/OOS split]
        MC[MonteCarlo\n1,000 shuffles]
        ST[Statistics\nt-test + bootstrap]
        MX[Metrics\nSharpe / Sortino / Calmar]
    end

    subgraph BrokerLayer["Brokers (src/brokers/)"]
        PB[PaperBroker\nvirtual positions]
        KB[KabuStation\nauKabucom REST]
    end

    subgraph APILayer["REST API (src/api/)"]
        SRV[FastAPI :8765]
        S_STAT[GET /api/status]
        S_POS[GET /api/positions]
        S_PERF[GET /api/performance]
        S_BT[POST /api/backtest]
    end

    subgraph Storage
        SQLite[(data_cache.db)]
    end

    YF --> YFC --> DM
    JQ --> JQC --> DM
    ED --> EDC --> DM
    DM <--> SQLite
    BASE --> MAC & DMO & MRS & BRA & LGB
    DM --> BT
    MAC & DMO & MRS & BRA & LGB --> BT
    BT --> WF & MC & ST & MX
    SRV --> S_STAT & S_POS & S_PERF & S_BT
    S_BT --> BT
    S_POS --> PB
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SPEC.md](docs/SPEC.md) | Full API specification |
| [docs/MANUAL.md](docs/MANUAL.md) | Setup and operations guide |
| [docs/TEST_LIST.md](docs/TEST_LIST.md) | Test case catalogue |
| [docs/ARCHITECTURE.mmd](docs/ARCHITECTURE.mmd) | Detailed Mermaid architecture diagram |

## Contributing

1. Fork the repo and create a feature branch (`git checkout -b feat/your-feature`)
2. Write tests first — maintain 80%+ coverage (`pytest --cov=src`)
3. Follow immutable data patterns: never mutate DataFrames in-place, return copies
4. Submit a pull request with a description of what and why

Please read [docs/SPEC.md](docs/SPEC.md) for design conventions before contributing.

## License

Apache 2.0 — see [LICENSE](LICENSE)
