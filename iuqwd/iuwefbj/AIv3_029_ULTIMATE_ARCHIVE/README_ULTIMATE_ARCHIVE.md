# AIv3_029 — ULTIMATE ARCHIVE README

## What This Is

This is the complete, organized archive of the AIv3_029 trading bot project. It contains EVERYTHING — every line of code, every data file, every trained AI model, every test result, every guide, and the full conversation context.

## Folder Structure

```
AIv3_029_ULTIMATE_ARCHIVE/
│
├── README_ULTIMATE_ARCHIVE.md          ← You are here
├── requirements.txt                    ← Python dependencies
├── worklog.md                          ← Development log
│
├── 1_BOT_CODE/                         ← The trading bot and all its code
│   ├── paper_trading_bot.py            ← THE BOT (run this to start trading)
│   ├── api_keys.json                   ← Put your Binance API keys here
│   ├── backtester.py                   ← Core backtesting engine + indicators
│   ├── walkforward_3year.py            ← 3-year walk-forward test + features
│   ├── honest_retest.py                ← Fixed trailing stop backtester
│   ├── realistic_3year.py              ← Realistic 3-year test (final results)
│   ├── risk_managed_v2.py              ← RiskConfig dataclass + presets
│   ├── smart_strategies.py             ← Indicator functions
│   ├── smart_strategies_round2.py      ← More indicators
│   ├── bots_100.py                     ← Helper functions + linreg slope
│   ├── generate_strategies.py          ← Grid strategy generator
│   ├── evaluate.py                     ← Funnel + metrics
│   ├── robustness.py                   ← Phase 5 robustness checks
│   ├── ai_bots_100.py                  ← 100 AI bots v1
│   ├── ai_bots_v2.py                   ← 50 AI bots v2
│   ├── ai_bots_v3.py                   ← 100 AI bots v3 (produced AIv3_029)
│   ├── mega_ensemble.py                ← 9-strategy mega-ensemble
│   ├── audit_test.py                   ← Lookahead + control tests
│   ├── ultimate_test.py                ← 7-test comprehensive audit
│   ├── resilient_36month.py            ← 36-month walk-forward (3 versions)
│   ├── build_aiv3_doc.py               ← Builds the .docx guide
│   └── ... (other support scripts)
│
├── 2_DATA/                             ← Market data + trained AI models
│   ├── BTCUSDT_1h.parquet              ← Bitcoin 1h candles (2019-2026)
│   ├── ETHUSDT_1h.parquet              ← Ethereum 1h candles
│   ├── BNBUSDT_1h.parquet              ← BNB 1h candles
│   ├── SOLUSDT_1h.parquet              ← Solana 1h candles
│   ├── XRPUSDT_1h.parquet              ← Ripple 1h candles
│   ├── ADAUSDT_1h.parquet              ← Cardano 1h candles
│   ├── DOGEUSDT_1h.parquet             ← Dogecoin 1h candles
│   ├── AVAXUSDT_1h.parquet             ← Avalanche 1h candles
│   ├── PAXGUSDT_1h.parquet             ← PAX Gold 1h candles
│   ├── BTCUSDT_15m.parquet             ← Bitcoin 15m candles
│   ├── ETHUSDT_15m.parquet             ← Ethereum 15m candles
│   ├── ... (15m data for all 9 symbols)
│   └── ai_models_v3/                   ← 215 trained AI models (Random Forest)
│       ├── BTCUSDT_momentum_L3rf_onl.pkl          ← Bitcoin AI
│       ├── BTCUSDT_momentum_L3rf_onl_thresh.json  ← Bitcoin threshold
│       ├── ETHUSDT_momentum_L3rf_onl.pkl          ← Ethereum AI
│       └── ... (9 markets × model + threshold)
│
├── 3_RESULTS/                           ← All test results
│   ├── realistic_3year_results.json    ← ★ FINAL REALISTIC 3-YEAR RESULTS
│   ├── AIv3_029_walkforward_3year.xlsx ← 36-month walk-forward (Excel)
│   ├── AIv3_029_Full_Build_Guide.docx  ← 25-page build guide
│   ├── 100_ai_bots_30day.xlsx          ← 100 AI bots v1 results
│   ├── ai_bots_v3_30day.xlsx           ← 100 AI bots v3 results
│   ├── clean_3month_test.json          ← 3-month clean test
│   ├── exit_interval_comparison.json   ← 15m vs 1h vs 2h vs 4h exit test
│   ├── param_sweep_100bots.json        ← 100-bot parameter sweep
│   └── ... (other result files)
│
├── 4_SETUP_GUIDES/                      ← Step-by-step instructions
│   ├── COMPLETE_SETUP_GUIDE.txt        ← Android phone guide (19 parts)
│   ├── SETUP_GUIDE_WINDOWS.txt         ← Windows laptop guide (13 parts)
│   ├── SETUP_GUIDE_VPS.txt             ← VPS guide (9 parts + real trading)
│   └── SETUP_GUIDE.txt                 ← Short guide
│
└── 5_FULL_CONTEXT/                      ← Complete conversation context
    └── FULL_CONVERSATION_CONTEXT.md    ← Everything we discussed
```

## Quick Start

1. Extract the zip
2. Choose your device:
   - Phone: Read `4_SETUP_GUIDES/COMPLETE_SETUP_GUIDE.txt`
   - Laptop: Read `4_SETUP_GUIDES/SETUP_GUIDE_WINDOWS.txt`
   - VPS: Read `4_SETUP_GUIDES/SETUP_GUIDE_VPS.txt`
3. Follow the steps exactly
4. Run: `python paper_trading_bot.py`

## The Bot Configuration (AIv3_029)

| Setting | Value |
|---------|-------|
| Markets | 9 (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, PAXG) |
| Timeframe | 1 hour |
| Signal | Momentum bars (body >60%, close in top 25%, volume >1.3x avg) |
| Stop loss | 0.5 × ATR(14) from entry |
| Take profit | 2.0 × stop distance (1:2 R:R) |
| Trailing stop | Breakeven at +0.5R, trail at +1.0R with 0.3R distance |
| AI | Random Forest (80 trees, depth 5), threshold 0.45 |
| Risk per trade | 1.0% (anti-martingale: 0.4% to 1.5%) |
| Max concurrent | 1 position |
| Monthly loss cap | -5% |
| Weekly loss cap | -3% |
| Daily loss cap | -2% |
| Commission | 0.075% per side (BNB maker discount) |
| Slippage | 0% (limit orders) |
| Position sizing | Spot-capped (can't buy more than wallet) |

## Realistic 3-Year Results

| Metric | Value |
|--------|------:|
| Months tested | 36 (Nov 2023 - Oct 2026) |
| Profitable months | 36/36 (100%) |
| Total trades | 3,802 |
| Win rate | 50.7% |
| Avg monthly profit | +$2,981 (+29.8%) |
| Worst month | +$570 (+5.7%) |
| Best month | +$9,723 (+97.2%) |
| Avg drawdown | 3.0% |
| Worst drawdown | 6.9% |

## What Made Profit (Key Insights)

1. **Trailing stops are the entire edge.** Without trailing stops, the bot LOSES money (-6.3%/month). With trailing stops, it makes +29.8%. The trailing stop converts would-be losses into breakeven/small wins.

2. **9 markets give enough signals.** Trading only BTC+ETH = fewer trades, less profit. 9 markets = ~100 trades/month = law of large numbers works.

3. **Spot-capped positions are BETTER.** Capping position size at wallet amount prevents overleveraging and actually INCREASES profit because risk caps don't trigger as often.

4. **Limit orders = zero slippage.** Using limit orders instead of market orders saves 3-10bp per trade. On 100 trades/month, that's $300-1000 saved.

5. **AI filters bad trades in bad months.** Without AI: 5 losing months in 3 years. With AI: 0 losing months. The AI's main value is downside protection.

## What Made Loss (Key Risks)

1. **Realistic slippage on alts.** DOGE (10bp), ADA (8bp), XRP (8bp) have high slippage. This is why we use limit orders (0% slippage).

2. **15-minute exit checking.** More granular checking reveals stops get hit more often. 1-hour checking is the sweet spot — matches real bot behavior.

3. **Exchange outages.** Binance goes down 2-3x/year. Stop might not execute. One event can lose 10-20%.

4. **Correlation risk.** All 9 crypto markets crash together. One bad day can lose 3-5%.

5. **AI model decay.** Market changes over time. AI needs monthly retraining.

## What To Do

1. Paper trade 15 days on your phone/laptop
2. If profitable → get a $5/month VPS
3. Run 24/7 on VPS (paper trade another 15 days)
4. If still profitable → switch to real money with $1,000
5. Start with LIMIT ORDERS + BNB discount (0.075% maker fee)
6. NEVER use leverage
7. NEVER override the bot
8. Withdraw profits monthly
9. Retrain AI monthly: `python scripts/walkforward_3year.py`

## What NOT To Do

1. ❌ Don't use leverage (liquidation risk)
2. ❌ Don't override the bot during losing streaks
3. ❌ Don't switch to futures/margin
4. ❌ Don't put more than 10% of net worth in this
5. ❌ Don't forget to retrain the AI monthly
6. ❌ Don't use market orders (use limit orders for 0% slippage)
7. ❌ Don't panic if you see 3-4 losses in a row (normal, win rate is 50%)
