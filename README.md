# DaiVietQuantCrypto Research

Repo nghiên cứu chiến lược giao dịch crypto sử dụng phương pháp định lượng.

## Cấu trúc dự án

```
DaiVietQuantCrypto_Research/
├── 📊 data/                    # Dữ liệu thô từ các sàn giao dịch
├── 🧠 strategies/              # Các chiến lược giao dịch chính
│   ├── momentum/               # Chiến lược momentum
│   ├── mean_reversion/         # Chiến lược mean reversion
│   ├── arbitrage/              # Chiến lược arbitrage
│   └── machine_learning/       # Chiến lược ML/AI
├── 📈 analysis/                # Phân tích và backtesting
├── 🛠️ utils/                   # Tiện ích và helper functions
├── 📋 research/                # Báo cáo nghiên cứu
└── 🚀 deployment/              # Triển khai chiến lược
```

## Theme chính: Chiến lược Momentum Multi-Asset

### Mục tiêu

- Phát triển chiến lược momentum đa tài sản
- Tối ưu hóa allocation dựa trên volatility và correlation
- Risk management tự động

### Child Themes

#### 1. Momentum Strategy

- **Mục tiêu**: Tận dụng xu hướng giá trong thị trường crypto
- **Phương pháp**: Technical indicators, trend following
- **Tài sản**: BTC, ETH, ADA, ALGO, ATA, BLZ

#### 2. Mean Reversion Strategy

- **Mục tiêu**: Giao dịch khi giá lệch khỏi trung bình
- **Phương pháp**: Bollinger Bands, RSI, statistical arbitrage
- **Tài sản**: Tất cả cặp tiền có sẵn

#### 3. Arbitrage Strategy

- **Mục tiêu**: Tận dụng chênh lệch giá giữa các sàn
- **Phương pháp**: Cross-exchange arbitrage, triangular arbitrage
- **Tài sản**: Các cặp tiền có tính thanh khoản cao

#### 4. Machine Learning Strategy

- **Mục tiêu**: Dự đoán giá sử dụng ML/AI
- **Phương pháp**: LSTM, Transformer, Ensemble methods
- **Tài sản**: Tất cả với focus vào BTC/ETH

## Dữ liệu

Repo hiện có dữ liệu từ Binance cho các cặp:

- BTC/USDT, ETH/USDT
- ADA/BTC, ADA/BUSD
- ACM/BTC, ACM/BUSD
- ALGO/USDT, ATA/USDT, BLZ/USDT
