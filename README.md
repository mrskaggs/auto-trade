# Alpaca Trading Bot - RSI Strategy

Autonomous cryptocurrency and stock trading bot using RSI (Relative Strength Index) strategy with Alpaca API.

## Features

- 🤖 **Automated RSI Trading** - Buys at oversold (RSI ≤30), sells at overbought (RSI ≥70)
- 💰 **Risk Management** - Dynamic position sizing based on portfolio percentage
- 📊 **Live Data** - Real-time crypto prices from CoinGecko + Yahoo Finance historical data
- 🔄 **Rate Limit Protection** - Smart caching and retry logic
- 🐳 **Docker Ready** - Deploy anywhere with Docker/Portainer
- 🔒 **Secure** - No hardcoded credentials, environment-based config

## Quick Start

### Local Development

1. **Clone Repository**
   ```bash
   git clone <your-repo>
   cd auto-trade
   ```

2. **Create Environment File**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` with your credentials**
   ```env
   APCA_API_KEY_ID=your_alpaca_key_here
   APCA_API_SECRET_KEY=your_alpaca_secret_here
   TRADING_SYMBOL=ETHUSD
   TRADING_RISK_PCT=1.0
   RSI_PERIOD=5
   RSI_BUY_THRESH=30
   RSI_SELL_THRESH=70
   ```

4. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run Bot**
   ```bash
   python realtime_trading_bot.py
   ```

## Docker Deployment

### Build & Run Locally

```bash
# Build image
docker build -t alpaca-trading-bot .

# Run with environment file
docker run -d --name trading-bot --env-file .env alpaca-trading-bot
```

### Deploy with Docker Compose

```bash
docker-compose up -d
```

## Portainer Deployment

### Method 1: Git Repository (Recommended)

1. **Push to GitHub** (make sure `.env` is in `.gitignore`!)
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **In Portainer:**
   - Go to **Stacks** → **Add Stack**
   - Name: `alpaca-trading-bot`
   - **Build method:** Git Repository
   - Repository URL: `https://github.com/yourusername/auto-trade`
   - Compose path: `docker-compose.yml`

3. **Add Environment Variables:**
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   APCA_API_BASE_URL=https://paper-api.alpaca.markets
   TRADING_SYMBOL=ETHUSD
   TRADING_RISK_PCT=1.0
   RSI_PERIOD=5
   RSI_BUY_THRESH=30
   RSI_SELL_THRESH=70
   ```

4. **Deploy Stack**

### Method 2: Web Editor

1. Copy contents of `docker-compose.yml`
2. In Portainer: **Stacks** → **Add Stack** → **Web editor**
3. Paste compose file
4. Add environment variables
5. Deploy

### Method 3: Upload

1. Create `.tar.gz` of project (excluding `.env`, `venv/`, etc.)
2. Upload to Portainer
3. Configure environment variables
4. Deploy

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `APCA_API_KEY_ID` | Alpaca API Key | `PKXXX...` |
| `APCA_API_SECRET_KEY` | Alpaca Secret Key | `8uVpXXX...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `APCA_API_BASE_URL` | `https://paper-api.alpaca.markets` | Paper or live trading |
| `TRADING_SYMBOL` | `BTCUSD` | Crypto: BTCUSD, ETHUSD, etc. Stock: AAPL, TSLA, etc. |
| `TRADING_RISK_PCT` | `1.0` | Risk per trade as % of portfolio |
| `RSI_PERIOD` | `14` | RSI calculation period (5=aggressive, 14=standard, 20=conservative) |
| `RSI_BUY_THRESH` | `30` | Buy when RSI ≤ this value |
| `RSI_SELL_THRESH` | `70` | Sell when RSI ≥ this value |

## Trading Strategy

### RSI (Relative Strength Index)

- **Oversold (RSI ≤30)**: Market dip → **BUY signal**
- **Overbought (RSI ≥70)**: Market rally → **SELL signal**
- **Neutral (30-70)**: **HOLD** - no action

### Position Management

- **One position at a time** - prevents over-leveraging
- **Dynamic sizing** - calculates quantity based on portfolio value and risk %
- **Paper trading** - safe testing with virtual money

### Example

With $100k portfolio and 1% risk:
- **Risk per trade**: $1,000
- **ETH @ $3,000**: Buy 0.333 ETH
- **BTC @ $90,000**: Buy 0.011 BTC

## Monitoring

### View Logs

```bash
# Docker
docker logs -f alpaca-trading-bot

# Docker Compose
docker-compose logs -f

# Portainer
Stacks → alpaca-trading-bot → Logs
```

### Expected Output

```
INFO:root:=== TRADING BOT SETTINGS ===
INFO:root:Symbol: ETHUSD
INFO:root:RSI Period: 5
INFO:root:Risk per Trade: 1.0%
INFO:root:Starting real-time trading bot...
INFO:root:[Data] ETHUSD - Fetched 50 5-minute bars
INFO:root:[Heartbeat] ETHUSD - Price: $3037.92 (live), RSI: 28.45 (-2.15)
INFO:root:✅ BUY 0.3312 ETHUSD @ $3037.92 ($999.58 risk)
```

## Security Best Practices

✅ **Never commit `.env` file** - included in `.gitignore`  
✅ **Use paper trading first** - test with virtual money  
✅ **Rotate API keys regularly** - regenerate in Alpaca dashboard  
✅ **Monitor container logs** - watch for errors/unexpected behavior  
✅ **Set resource limits** - defined in docker-compose.yml  

## Troubleshooting

### Bot won't start

- Check environment variables are set correctly
- Verify Alpaca API keys are valid
- Check logs: `docker logs alpaca-trading-bot`

### No trades executing

- RSI must be ≤30 to buy (market dip)
- RSI must be ≥70 to sell (market rally)
- Must have cash to buy, position to sell
- Check logs for signals: `[Heartbeat]` shows current RSI

### Yahoo Finance errors

- Bot caches data for 30s to avoid rate limits
- Retries 3 times with 2s delays
- Falls back to cached data if all fail

## Supported Assets

### Crypto (24/7 Trading)
- BTCUSD (Bitcoin)
- ETHUSD (Ethereum)
- SOLUSD (Solana)
- ADAUSD (Cardano)
- DOGEUSD (Dogecoin)

### Stocks (Market Hours Only)
- AAPL, TSLA, GOOGL, MSFT, etc.

## Project Structure

```
auto-trade/
├── realtime_trading_bot.py   # Main bot code
├── Dockerfile                 # Container image
├── docker-compose.yml         # Stack configuration
├── requirements.txt           # Python dependencies
├── .env.example              # Example environment file
├── .gitignore                # Prevents committing secrets
├── .dockerignore             # Excludes files from image
└── README.md                 # This file
```

## Development

### Run Tests

```bash
python -m pytest tests/
```

### Format Code

```bash
black realtime_trading_bot.py
```

## License

MIT

## Disclaimer

⚠️ **Trading involves risk.** This bot is for educational purposes. Always test with paper trading first. Past performance does not guarantee future results. Use at your own risk.

## Support

For issues or questions, open a GitHub issue or contact the maintainer.
