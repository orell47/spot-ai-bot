# Spot AI Bot v0.4 — Paper Trading Challenge

This version is designed for a **$200 paper account** and a 30–60 day challenge toward a $2,000 stretch target. The target is aspirational; the bot must never increase risk simply to chase it.

## User-defined operating philosophy
- Spot only.
- Start with $200 paper equity.
- Normal position sizing around 10–15% of equity.
- Exceptional setups can use up to 25% of equity, never the whole account.
- Up to 10 open positions.
- Balanced strategy that can become aggressive when setup quality is unusually high.
- 50%+ recent price-move rejection remains a hard filter.
- Small-cap coins are allowed only through strict liquidity/order-book/manipulation filters.
- News and market events matter; missing/unclear information should reduce confidence.
- Hard kill switch for drawdown/daily loss and operational anomalies.
- All analyzed candidates, including rejected ones, are saved for later research/calibration.
- Learning may calibrate thresholds after enough closed trades, but cannot change safety rules.

## Important
This is **not live trading**. Live trading remains disabled. Do not add exchange trading API keys.

The current "AI" label refers to the decision engine/data pipeline; this release is primarily a rules + adaptive-calibration system, not a trained neural network. ML/modeling is a later stage after enough clean outcomes exist.

## What v0.4 adds
- Persistent $200 paper portfolio.
- Cash accounting.
- Open-position accounting.
- Stop loss and volatility-based trailing stop.
- Partial take profit and final take profit.
- Paper PnL in USD and percent.
- Equity snapshots and equity peak.
- Drawdown tracking.
- Daily realized PnL tracking.
- 50% total drawdown hard kill switch.
- Daily loss kill switch.
- Total exposure cap and cash reserve.
- 5-minute fast confirmation + 15-minute primary analysis.
- Adaptive score threshold after 30 closed trades.
- Telegram alerts for buys, exits, scans and risk status.

## Caveats
- News is RSS-based and is not yet an authoritative-source/event-verification layer.
- Token unlocks, on-chain holder concentration, contract verification and advanced scam detection are planned next.
- Paper execution is still an approximation of real exchange fills and slippage.
- GitHub scheduled jobs are not a low-latency trading environment.

## GitHub
The workflow runs one scan and then saves `data/bot.db` back into the repository. The repository should be public if using the free public-runner approach. That means the strategy source and paper journal are visible; secrets remain in GitHub Secrets and are not committed.

## Telegram secrets
Repository Settings → Secrets and variables → Actions:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Next gates before real money
1. Collect enough paper observations.
2. Implement/validate token unlock and on-chain risk layers.
3. Backtest and walk-forward test.
4. Validate paper results across different market regimes.
5. Demo trading.
6. Only then consider a tiny live Spot allocation, with the user explicitly enabling it.
