#!/usr/bin/env python3
"""
Trend Following Analysis for BTCUSDT Data
Import và phân tích dữ liệu từ Binance_BTCUSDT_d.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style cho plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrendFollowingAnalyzer:
    """Class để phân tích trend following cho BTCUSDT"""
    
    def __init__(self, data_path="data/Binance_BTCUSDT_d.csv"):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Load dữ liệu từ CSV file"""
        try:
            print("🔄 Đang load dữ liệu từ:", self.data_path)
            
            # Load data với header=None để bỏ qua header đặc biệt
            self.data = pd.read_csv(self.data_path, header=None)
            
            # Bỏ qua dòng đầu tiên (header của CryptoDataDownload)
            if self.data.iloc[0, 0] == 'https://www.CryptoDataDownload.com':
                self.data = self.data.iloc[1:].reset_index(drop=True)
            
            # Đặt tên cột theo header thực tế
            self.data.columns = ['Unix', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USDT', 'tradecount']
            
            # Convert Unix timestamp to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Set Date làm index
            self.data.set_index('Date', inplace=True)
            
            # Convert các cột số
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USDT', 'tradecount']
            for col in numeric_columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            print(f"✅ Đã load thành công {len(self.data)} records")
            print(f"📅 Date range: {self.data.index.min()} đến {self.data.index.max()}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Lỗi khi load dữ liệu: {e}")
            return None
    
    def display_data_info(self):
        """Hiển thị thông tin chung về dữ liệu"""
        if self.data is None:
            print("❌ Chưa có dữ liệu. Hãy load data trước.")
            return
        
        print("\n" + "="*60)
        print("📊 THÔNG TIN DỮ LIỆU BTCUSDT")
        print("="*60)
        
        # Basic info
        print(f"📈 Tổng số records: {len(self.data):,}")
        print(f"📅 Thời gian: {self.data.index.min().strftime('%Y-%m-%d')} đến {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"📊 Số ngày giao dịch: {(self.data.index.max() - self.data.index.min()).days} ngày")
        
        # Price statistics
        print(f"\n💰 THỐNG KÊ GIÁ:")
        print(f"   Giá cao nhất: ${self.data['High'].max():,.2f}")
        print(f"   Giá thấp nhất: ${self.data['Low'].min():,.2f}")
        print(f"   Giá hiện tại: ${self.data['Close'].iloc[-1]:,.2f}")
        print(f"   Giá trung bình: ${self.data['Close'].mean():,.2f}")
        
        # Volume statistics
        print(f"\n📊 THỐNG KÊ KHỐI LƯỢNG:")
        print(f"   Tổng volume BTC: {self.data['Volume BTC'].sum():,.2f} BTC")
        print(f"   Tổng volume USDT: ${self.data['Volume USDT'].sum():,.0f}")
        print(f"   Volume trung bình/ngày: {self.data['Volume BTC'].mean():,.2f} BTC")
        
        # Trade count
        print(f"\n🔄 THỐNG KÊ GIAO DỊCH:")
        print(f"   Tổng số trades: {self.data['tradecount'].sum():,}")
        print(f"   Trades trung bình/ngày: {self.data['tradecount'].mean():,.0f}")
        
        # Missing data
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n⚠️  DỮ LIỆU THIẾU:")
            for col, count in missing_data.items():
                if count > 0:
                    print(f"   {col}: {count} records")
        else:
            print(f"\n✅ Không có dữ liệu thiếu")
    
    def calculate_returns(self):
        """Tính toán returns và volatility"""
        if self.data is None:
            print("❌ Chưa có dữ liệu. Hãy load data trước.")
            return
        
        print("\n📈 TÍNH TOÁN RETURNS VÀ VOLATILITY")
        print("-" * 40)
        
        # Daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # Cumulative returns
        self.data['Cumulative_Return'] = (1 + self.data['Daily_Return']).cumprod()
        
        # Rolling volatility (30 days)
        self.data['Volatility_30d'] = self.data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
        
        # Statistics
        total_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (365 / len(self.data)) - 1
        volatility = self.data['Daily_Return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        print(f"📊 Tổng return: {total_return:.2%}")
        print(f"📈 Annualized return: {annualized_return:.2%}")
        print(f"📉 Volatility (annualized): {volatility:.2%}")
        print(f"📊 Sharpe ratio: {sharpe_ratio:.2f}")
        
        # Max drawdown
        cumulative = self.data['Cumulative_Return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        print(f"📉 Max drawdown: {max_drawdown:.2%}")
    
    def identify_trends(self):
        """Nhận diện trends trong dữ liệu"""
        if self.data is None:
            print("❌ Chưa có dữ liệu. Hãy load data trước.")
            return
        
        print("\n🔍 PHÂN TÍCH TREND")
        print("-" * 30)
        
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        # Trend signals
        self.data['Trend_20'] = np.where(self.data['Close'] > self.data['MA_20'], 'Uptrend', 'Downtrend')
        self.data['Trend_50'] = np.where(self.data['Close'] > self.data['MA_50'], 'Uptrend', 'Downtrend')
        self.data['Trend_200'] = np.where(self.data['Close'] > self.data['MA_200'], 'Uptrend', 'Downtrend')
        
        # Current trend status
        current_price = self.data['Close'].iloc[-1]
        ma_20 = self.data['MA_20'].iloc[-1]
        ma_50 = self.data['MA_50'].iloc[-1]
        ma_200 = self.data['MA_200'].iloc[-1]
        
        print(f"💰 Giá hiện tại: ${current_price:,.2f}")
        print(f"📊 MA 20: ${ma_20:,.2f} ({'📈 Above' if current_price > ma_20 else '📉 Below'})")
        print(f"📊 MA 50: ${ma_50:,.2f} ({'📈 Above' if current_price > ma_50 else '📉 Below'})")
        print(f"📊 MA 200: ${ma_200:,.2f} ({'📈 Above' if current_price > ma_200 else '📉 Below'})")
        
        # Trend strength
        uptrend_days = len(self.data[self.data['Trend_20'] == 'Uptrend'])
        downtrend_days = len(self.data[self.data['Trend_20'] == 'Downtrend'])
        
        print(f"\n📈 Uptrend days: {uptrend_days} ({uptrend_days/len(self.data)*100:.1f}%)")
        print(f"📉 Downtrend days: {downtrend_days} ({downtrend_days/len(self.data)*100:.1f}%)")
    
    def plot_data(self):
        """Vẽ biểu đồ dữ liệu"""
        if self.data is None:
            print("❌ Chưa có dữ liệu. Hãy load data trước.")
            return
        
        print("\n📊 Đang tạo biểu đồ...")
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price chart with moving averages
        axes[0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0].plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.7)
        axes[0].plot(self.data.index, self.data['MA_50'], label='MA 50', alpha=0.7)
        axes[0].plot(self.data.index, self.data['MA_200'], label='MA 200', alpha=0.7)
        axes[0].set_title('BTCUSDT Price Chart with Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume chart
        axes[1].bar(self.data.index, self.data['Volume BTC'], alpha=0.6, color='orange')
        axes[1].set_title('Daily Trading Volume (BTC)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volume (BTC)')
        axes[1].grid(True, alpha=0.3)
        
        # Returns and volatility
        axes[2].plot(self.data.index, self.data['Cumulative_Return'], label='Cumulative Returns', linewidth=2)
        axes[2].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Cumulative Returns')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/btcusdt_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Biểu đồ đã được lưu tại: results/btcusdt_analysis.png")
        plt.show()
    
    def run_analysis(self):
        """Chạy toàn bộ phân tích"""
        print("🚀 BẮT ĐẦU PHÂN TÍCH TREND FOLLOWING CHO BTCUSDT")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        if self.data is not None:
            # Display info
            self.display_data_info()
            
            # Calculate returns
            self.calculate_returns()
            
            # Identify trends
            self.identify_trends()
            
            # Plot data
            self.plot_data()
            
            print("\n✅ PHÂN TÍCH HOÀN THÀNH!")
        else:
            print("❌ Không thể load dữ liệu. Vui lòng kiểm tra file path.")


def main():
    """Main function"""
    analyzer = TrendFollowingAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 