import requests
import pandas as pd
import json
from typing import Dict, Optional, List, Tuple
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from urllib.parse import quote
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

@dataclass
class MarketData:
    """Enhanced data class to store market information"""
    area_code: str
    city: str = ""
    state: str = ""
    average_home_value: Optional[float] = None
    home_value_growth_yoy: Optional[float] = None
    for_sale_inventory: Optional[int] = None
    home_price_forecast: Optional[float] = None
    home_value_growth_5yr: Optional[float] = None
    price_cut_percentage: Optional[float] = None
    overvalued_percentage: Optional[float] = None
    cap_rate: Optional[float] = None
    median_income: Optional[float] = None
    rent_estimate: Optional[float] = None
    # New fields for enhanced analysis
    investment_score: Optional[float] = None
    risk_level: Optional[str] = None
    investment_grade: Optional[str] = None
    predicted_growth_1yr: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class InvestmentScoreComponents:
    """Breakdown of investment score components"""
    cash_flow_score: float = 0.0
    appreciation_score: float = 0.0
    market_stability_score: float = 0.0
    affordability_score: float = 0.0
    liquidity_score: float = 0.0
    total_score: float = 0.0
    grade: str = "C"
    risk_level: str = "Medium"

class InvestmentScoreCalculator:
    """Calculate comprehensive investment scores for real estate markets"""
    
    def __init__(self):
        # Scoring weights (should sum to 1.0)
        self.weights = {
            'cash_flow': 0.25,      # Cap rate, rent-to-price ratio
            'appreciation': 0.25,    # Historical and predicted growth
            'market_stability': 0.20, # Inventory, price cuts, volatility
            'affordability': 0.15,   # Price-to-income ratio
            'liquidity': 0.15       # Market size, transaction volume
        }
    
    def calculate_cash_flow_score(self, market_data: MarketData) -> float:
        """Score based on rental yields and cash flow potential"""
        score = 0.0
        
        # Cap rate scoring (0-100)
        if market_data.cap_rate:
            if market_data.cap_rate >= 8.0:
                score += 100
            elif market_data.cap_rate >= 6.0:
                score += 80
            elif market_data.cap_rate >= 4.0:
                score += 60
            elif market_data.cap_rate >= 2.0:
                score += 40
            else:
                score += 20
        else:
            score += 50  # Default moderate score
            
        return min(100, max(0, score))
    
    def calculate_appreciation_score(self, market_data: MarketData) -> float:
        """Score based on historical and predicted price appreciation"""
        score = 0.0
        
        # YoY growth scoring
        if market_data.home_value_growth_yoy is not None:
            yoy_growth = market_data.home_value_growth_yoy
            if yoy_growth >= 8.0:
                score += 45
            elif yoy_growth >= 5.0:
                score += 40
            elif yoy_growth >= 3.0:
                score += 35
            elif yoy_growth >= 0:
                score += 25
            else:
                score += 10
        
        # 5-year growth scoring
        if market_data.home_value_growth_5yr is not None:
            five_yr_growth = market_data.home_value_growth_5yr
            if five_yr_growth >= 40.0:
                score += 45
            elif five_yr_growth >= 25.0:
                score += 40
            elif five_yr_growth >= 15.0:
                score += 35
            elif five_yr_growth >= 0:
                score += 25
            else:
                score += 10
        
        # Price forecast scoring
        if market_data.home_price_forecast is not None:
            if market_data.home_price_forecast >= 70:
                score += 10
            elif market_data.home_price_forecast >= 55:
                score += 8
            elif market_data.home_price_forecast >= 45:
                score += 5
            else:
                score += 2
        
        return min(100, max(0, score))
    
    def calculate_market_stability_score(self, market_data: MarketData) -> float:
        """Score based on market stability indicators"""
        score = 50.0  # Start with neutral
        
        # Inventory levels (lower is better for stability)
        if market_data.for_sale_inventory:
            if market_data.for_sale_inventory < 500:
                score += 25  # Low inventory = high stability
            elif market_data.for_sale_inventory < 1000:
                score += 15
            elif market_data.for_sale_inventory < 2000:
                score += 5
            else:
                score -= 10  # High inventory = potential instability
        
        # Price cuts (lower is better)
        if market_data.price_cut_percentage is not None:
            if market_data.price_cut_percentage < 5:
                score += 25
            elif market_data.price_cut_percentage < 10:
                score += 15
            elif market_data.price_cut_percentage < 15:
                score += 5
            else:
                score -= 10
        
        return min(100, max(0, score))
    
    def calculate_affordability_score(self, market_data: MarketData) -> float:
        """Score based on affordability metrics"""
        score = 50.0  # Start neutral
        
        # Overvalued percentage (lower is better)
        if market_data.overvalued_percentage is not None:
            if market_data.overvalued_percentage < -10:
                score += 50  # Undervalued = great score
            elif market_data.overvalued_percentage < 0:
                score += 30
            elif market_data.overvalued_percentage < 10:
                score += 10
            elif market_data.overvalued_percentage < 25:
                score -= 10
            else:
                score -= 30  # Significantly overvalued
        
        return min(100, max(0, score))
    
    def calculate_liquidity_score(self, market_data: MarketData) -> float:
        """Score based on market liquidity (ability to buy/sell quickly)"""
        score = 50.0  # Start neutral
        
        # Major market bonus
        major_markets = ['New York', 'Los Angeles', 'Chicago', 'Houston', 
                        'Philadelphia', 'Phoenix', 'San Diego', 'Dallas', 
                        'San Jose', 'Austin', 'Charlotte', 'Atlanta', 'Miami', 'Seattle']
        
        if market_data.city in major_markets:
            score += 30
        
        # Inventory as liquidity proxy (moderate inventory is best)
        if market_data.for_sale_inventory:
            if 800 <= market_data.for_sale_inventory <= 1500:
                score += 20  # Goldilocks zone
            elif 500 <= market_data.for_sale_inventory <= 2000:
                score += 10
            else:
                score += 5
        
        return min(100, max(0, score))
    
    def calculate_investment_score(self, market_data: MarketData) -> InvestmentScoreComponents:
        """Calculate comprehensive investment score"""
        
        # Calculate component scores
        cash_flow_score = self.calculate_cash_flow_score(market_data)
        appreciation_score = self.calculate_appreciation_score(market_data)
        market_stability_score = self.calculate_market_stability_score(market_data)
        affordability_score = self.calculate_affordability_score(market_data)
        liquidity_score = self.calculate_liquidity_score(market_data)
        
        # Calculate weighted total score
        total_score = (
            cash_flow_score * self.weights['cash_flow'] +
            appreciation_score * self.weights['appreciation'] +
            market_stability_score * self.weights['market_stability'] +
            affordability_score * self.weights['affordability'] +
            liquidity_score * self.weights['liquidity']
        )
        
        # Determine grade and risk level
        if total_score >= 85:
            grade = "A+"
            risk_level = "Low"
        elif total_score >= 80:
            grade = "A"
            risk_level = "Low"
        elif total_score >= 75:
            grade = "A-"
            risk_level = "Low-Medium"
        elif total_score >= 70:
            grade = "B+"
            risk_level = "Medium"
        elif total_score >= 65:
            grade = "B"
            risk_level = "Medium"
        elif total_score >= 60:
            grade = "B-"
            risk_level = "Medium"
        elif total_score >= 55:
            grade = "C+"
            risk_level = "Medium-High"
        elif total_score >= 50:
            grade = "C"
            risk_level = "High"
        elif total_score >= 45:
            grade = "C-"
            risk_level = "High"
        else:
            grade = "D"
            risk_level = "Very High"
        
        return InvestmentScoreComponents(
            cash_flow_score=cash_flow_score,
            appreciation_score=appreciation_score,
            market_stability_score=market_stability_score,
            affordability_score=affordability_score,
            liquidity_score=liquidity_score,
            total_score=total_score,
            grade=grade,
            risk_level=risk_level
        )

class DataVisualizer:
    """Enhanced data visualization for market analysis"""
    
    def __init__(self):
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_investment_dashboard(self, market_data_list: List[MarketData], 
                                  save_path: str = None) -> None:
        """Create comprehensive investment dashboard"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Prepare data
        df = pd.DataFrame([asdict(md) for md in market_data_list])
        df['location'] = df['city'] + ', ' + df['state']
        
        # 1. Investment Scores Ranking (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'investment_score' in df.columns:
            top_markets = df.nlargest(10, 'investment_score')
            bars = ax1.barh(range(len(top_markets)), top_markets['investment_score'])
            ax1.set_yticks(range(len(top_markets)))
            ax1.set_yticklabels(top_markets['location'])
            ax1.set_xlabel('Investment Score')
            ax1.set_title('Top Investment Markets by Score', fontsize=14, fontweight='bold')
            
            # Color bars by grade
            for i, (_, row) in enumerate(top_markets.iterrows()):
                if row['investment_score'] >= 80:
                    bars[i].set_color('green')
                elif row['investment_score'] >= 70:
                    bars[i].set_color('orange')
                else:
                    bars[i].set_color('red')
        
        # 2. Cap Rate vs Home Value (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'cap_rate' in df.columns and 'average_home_value' in df.columns:
            scatter = ax2.scatter(df['average_home_value']/1000, df['cap_rate'], 
                                c=df['investment_score'], cmap='RdYlGn', s=100, alpha=0.7)
            ax2.set_xlabel('Average Home Value ($000s)')
            ax2.set_ylabel('Cap Rate (%)')
            ax2.set_title('Cap Rate vs Home Value', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax2, label='Investment Score')
            
            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df['average_home_value']/1000, df['cap_rate'], 1)
                p = np.poly1d(z)
                ax2.plot(df['average_home_value']/1000, p(df['average_home_value']/1000), 
                        "r--", alpha=0.8)
        
        # 3. Growth Trends (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'home_value_growth_yoy' in df.columns:
            df_sorted = df.sort_values('home_value_growth_yoy', ascending=True)
            bars = ax3.barh(range(len(df_sorted)), df_sorted['home_value_growth_yoy'])
            ax3.set_yticks(range(len(df_sorted)))
            ax3.set_yticklabels(df_sorted['location'])
            ax3.set_xlabel('YoY Growth (%)')
            ax3.set_title('Home Value Growth (Year-over-Year)', fontsize=14, fontweight='bold')
            
            # Color by growth rate
            for i, growth in enumerate(df_sorted['home_value_growth_yoy']):
                if growth >= 6:
                    bars[i].set_color('darkgreen')
                elif growth >= 3:
                    bars[i].set_color('lightgreen')
                elif growth >= 0:
                    bars[i].set_color('yellow')
                else:
                    bars[i].set_color('red')
        
        # 4. Price Cut Analysis (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'price_cut_percentage' in df.columns and 'for_sale_inventory' in df.columns:
            scatter = ax4.scatter(df['for_sale_inventory'], df['price_cut_percentage'], 
                                c=df['investment_score'], cmap='RdYlGn', s=100, alpha=0.7)
            ax4.set_xlabel('For Sale Inventory')
            ax4.set_ylabel('Price Cut Percentage (%)')
            ax4.set_title('Market Stress Indicators', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='Investment Score')
        
        # 5. Risk vs Return Matrix (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'cap_rate' in df.columns and 'home_value_growth_yoy' in df.columns:
            # Create risk score (inverse of investment score)
            df['risk_score'] = 100 - df['investment_score']
            
            scatter = ax5.scatter(df['risk_score'], df['cap_rate'], 
                                s=df['home_value_growth_yoy']*10, alpha=0.6)
            ax5.set_xlabel('Risk Score (Higher = More Risky)')
            ax5.set_ylabel('Cap Rate (%)')
            ax5.set_title('Risk vs Return Analysis\n(Bubble size = YoY Growth)', fontsize=14, fontweight='bold')
            
            # Add quadrant lines
            ax5.axhline(y=df['cap_rate'].median(), color='gray', linestyle='--', alpha=0.5)
            ax5.axvline(x=df['risk_score'].median(), color='gray', linestyle='--', alpha=0.5)
        
        # 6. Overvaluation Heatmap (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'overvalued_percentage' in df.columns:
            # Create heatmap data
            overval_data = df[['location', 'overvalued_percentage', 'investment_score']].copy()
            overval_data = overval_data.sort_values('overvalued_percentage')
            
            # Create color map
            colors = []
            for val in overval_data['overvalued_percentage']:
                if val < -10:
                    colors.append('darkgreen')
                elif val < 0:
                    colors.append('lightgreen')
                elif val < 10:
                    colors.append('yellow')
                elif val < 25:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars = ax6.barh(range(len(overval_data)), overval_data['overvalued_percentage'], 
                           color=colors)
            ax6.set_yticks(range(len(overval_data)))
            ax6.set_yticklabels(overval_data['location'])
            ax6.set_xlabel('Overvaluation (%)')
            ax6.set_title('Market Valuation Analysis', fontsize=14, fontweight='bold')
            ax6.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        
        # 7. Investment Score Components Radar Chart (Bottom)
        ax7 = fig.add_subplot(gs[3, :])
        if len(market_data_list) > 0:
            # Get top 5 markets for radar comparison
            top_5_markets = df.nlargest(5, 'investment_score')
            
            # Create sample component scores for radar chart
            components = ['Cash Flow', 'Appreciation', 'Stability', 'Affordability', 'Liquidity']
            
            # Plot text summary instead of complex radar
            summary_text = "📊 INVESTMENT ANALYSIS SUMMARY:\n\n"
            
            for _, market in top_5_markets.iterrows():
                grade = market.get('investment_grade', 'N/A')
                score = market.get('investment_score', 0)
                risk = market.get('risk_level', 'Unknown')
                
                summary_text += f"🏆 {market['location']}: Grade {grade} | Score: {score:.1f} | Risk: {risk}\n"
                summary_text += f"    💰 Cap Rate: {market.get('cap_rate', 0):.2f}% | "
                summary_text += f"📈 YoY Growth: {market.get('home_value_growth_yoy', 0):.1f}%\n\n"
            
            ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.axis('off')
        
        # Main title
        fig.suptitle('🏠 REMarketIntel Investment Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_market_comparison_chart(self, market_data_list: List[MarketData], 
                                     save_path: str = None) -> None:
        """Create side-by-side market comparison"""
        
        if len(market_data_list) < 2:
            print("⚠️  Need at least 2 markets for comparison")
            return
        
        # Prepare data
        df = pd.DataFrame([asdict(md) for md in market_data_list])
        df['location'] = df['city'] + ', ' + df['state']
        
        # Create comparison figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Investment Scores
        bars1 = ax1.bar(df['location'], df['investment_score'])
        ax1.set_title('Investment Scores Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Investment Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color bars by score
        for bar, score in zip(bars1, df['investment_score']):
            if score >= 80:
                bar.set_color('green')
            elif score >= 70:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 2. Key Metrics Comparison
        metrics = ['cap_rate', 'home_value_growth_yoy', 'price_cut_percentage']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax2.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())
        
        ax2.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Markets')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(df['location'], rotation=45)
        ax2.legend()
        
        # 3. Risk vs Return
        if 'cap_rate' in df.columns and 'investment_score' in df.columns:
            scatter = ax3.scatter(100 - df['investment_score'], df['cap_rate'], 
                                s=200, c=df['home_value_growth_yoy'], cmap='RdYlGn')
            ax3.set_xlabel('Risk Score (Higher = More Risk)')
            ax3.set_ylabel('Cap Rate (%)')
            ax3.set_title('Risk vs Return Matrix', fontsize=14, fontweight='bold')
            
            # Add market labels
            for i, txt in enumerate(df['city']):
                ax3.annotate(txt, (100 - df['investment_score'].iloc[i], df['cap_rate'].iloc[i]))
            
            plt.colorbar(scatter, ax=ax3, label='YoY Growth %')
        
        # 4. Market Health Score
        health_metrics = ['for_sale_inventory', 'overvalued_percentage', 'price_cut_percentage']
        health_data = df[['location'] + [m for m in health_metrics if m in df.columns]]
        
        # Normalize for comparison (inverse some metrics)
        normalized_data = health_data.copy()
        if 'for_sale_inventory' in normalized_data.columns:
            normalized_data['inventory_score'] = 100 - ((normalized_data['for_sale_inventory'] - normalized_data['for_sale_inventory'].min()) / 
                                                        (normalized_data['for_sale_inventory'].max() - normalized_data['for_sale_inventory'].min()) * 100)
        
        ax4.bar(df['location'], df.get('investment_score', 50))
        ax4.set_title('Overall Market Health', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Health Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Market Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

class MLPredictor:
    """Machine Learning predictor for real estate market trends"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, market_data_list: List[MarketData]) -> pd.DataFrame:
        """Prepare features for ML model"""
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(md) for md in market_data_list])
        
        # Select numeric features for prediction
        feature_columns = [
            'average_home_value', 'home_value_growth_yoy', 'for_sale_inventory',
            'home_price_forecast', 'home_value_growth_5yr', 'price_cut_percentage',
            'overvalued_percentage', 'cap_rate', 'median_income', 'rent_estimate'
        ]
        
        # Keep only available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Fill missing values with median
        feature_df = df[available_features].copy()
        for col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Add derived features
        if 'average_home_value' in feature_df.columns and 'median_income' in feature_df.columns:
            feature_df['price_to_income_ratio'] = feature_df['average_home_value'] / feature_df['median_income']
        
        if 'rent_estimate' in feature_df.columns and 'average_home_value' in feature_df.columns:
            feature_df['rent_to_price_ratio'] = (feature_df['rent_estimate'] * 12) / feature_df['average_home_value']
        
        if 'for_sale_inventory' in feature_df.columns and 'average_home_value' in feature_df.columns:
            feature_df['inventory_to_value_ratio'] = feature_df['for_sale_inventory'] / (feature_df['average_home_value'] / 100000)
        
        self.feature_names = list(feature_df.columns)
        return feature_df
    
    def train_model(self, market_data_list: List[MarketData]) -> Dict:
        """Train ML model to predict market trends"""
        
        if len(market_data_list) < 10:
            print("⚠️  Need at least 10 markets for reliable ML training")
            return {'success': False, 'message': 'Insufficient data'}
        
        try:
            # Prepare features
            feature_df = self.prepare_features(market_data_list)
            
            if feature_df.empty:
                return {'success': False, 'message': 'No valid features found'}
            
            # Create synthetic historical data for training (in production, use real historical data)
            X, y = self.create_synthetic_training_data(feature_df)
            
            if len(X) < 20:
                return {'success': False, 'message': 'Insufficient training data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            return {
                'success': True,
                'mae': mae,
                'r2_score': r2,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Training error: {str(e)}'}
    
    def create_synthetic_training_data(self, current_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic historical data for training (replace with real data in production)"""
        
        synthetic_data = []
        synthetic_targets = []
        
        # For each current market, create variations representing historical states
        for _, row in current_df.iterrows():
            for i in range(10):  # Create 10 variations per market
                # Add random variations to simulate historical data
                variation = row.copy()
                
                # Add realistic noise and trends
                for col in variation.index:
                    if col in ['home_value_growth_yoy', 'home_value_growth_5yr']:
                        # Growth rates can vary significantly
                        variation[col] = variation[col] + np.random.normal(0, 2)
                    elif col in ['price_cut_percentage', 'overvalued_percentage']:
                        # These can have wider variations
                        variation[col] = variation[col] + np.random.normal(0, 5)
                    elif col in ['for_sale_inventory']:
                        # Inventory can vary significantly
                        variation[col] = max(100, variation[col] + np.random.normal(0, 200))
                    else:
                        # Other features with smaller variations
                        variation[col] = variation[col] + np.random.normal(0, variation[col] * 0.1)
                
                synthetic_data.append(variation.values)
                
                # Create target (future growth) based on current indicators
                target_growth = self.calculate_target_growth(variation)
                synthetic_targets.append(target_growth)
        
        return np.array(synthetic_data), np.array(synthetic_targets)
    
    def calculate_target_growth(self, features: pd.Series) -> float:
        """Calculate target growth based on market fundamentals"""
        
        base_growth = 3.0  # Base market growth
        
        # Adjust based on various factors
        if 'home_price_forecast' in features.index:
            forecast_factor = (features['home_price_forecast'] - 50) / 10
            base_growth += forecast_factor
        
        if 'overvalued_percentage' in features.index:
            # Overvalued markets tend to have lower future growth
            overval_factor = -features['overvalued_percentage'] / 10
            base_growth += overval_factor
        
        if 'for_sale_inventory' in features.index:
            # High inventory suggests lower growth
            inventory_factor = -max(0, (features['for_sale_inventory'] - 1000) / 500)
            base_growth += inventory_factor
        
        if 'cap_rate' in features.index:
            # Higher cap rates suggest better fundamentals
            cap_factor = (features['cap_rate'] - 5) / 2
            base_growth += cap_factor
        
        # Add some randomness to simulate market volatility
        base_growth += np.random.normal(0, 1)
        
        return base_growth
    
    def predict_market_growth(self, market_data: MarketData) -> Dict:
        """Predict future market growth for a single market"""
        
        if not self.is_trained:
            return {'success': False, 'message': 'Model not trained yet'}
        
        try:
            # Prepare single market features
            df = pd.DataFrame([asdict(market_data)])
            feature_df = self.prepare_features([market_data])
            
            if feature_df.empty:
                return {'success': False, 'message': 'Could not extract features'}
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence interval using prediction std
            predictions_ensemble = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(features_scaled)[0]
                predictions_ensemble.append(pred)
            
            pred_std = np.std(predictions_ensemble)
            confidence_lower = prediction - 1.96 * pred_std
            confidence_upper = prediction + 1.96 * pred_std
            
            return {
                'success': True,
                'predicted_growth_1yr': prediction,
                'confidence_interval': (confidence_lower, confidence_upper),
                'confidence_std': pred_std,
                'model_confidence': 'High' if pred_std < 1.5 else 'Medium' if pred_std < 3.0 else 'Low'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Prediction error: {str(e)}'}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        
        if not self.is_trained:
            return {}
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

class EnhancedRealEstateMarketResearcher:
    """Enhanced version with investment scoring, visualization, and ML predictions"""
    
    def __init__(self):
        # Initialize base researcher (keeping all original functionality)
        self.rentcast_api_key = os.getenv('RENTCAST_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        if not self.rentcast_api_key:
            raise ValueError("RENTCAST_API_KEY not found in .env file")
        
        self.rentcast_base_url = "https://api.rentcast.io/v1"
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        
        self.headers = {
            'X-Api-Key': self.rentcast_api_key,
            'accept': 'application/json'
        }
        
        # Area code to location mapping (same as original)
        self.area_code_locations = {
            '212': {'city': 'New York', 'state': 'NY', 'zip': '10001'},
            '310': {'city': 'Los Angeles', 'state': 'CA', 'zip': '90210'},
            '312': {'city': 'Chicago', 'state': 'IL', 'zip': '60601'},
            '713': {'city': 'Houston', 'state': 'TX', 'zip': '77001'},
            '215': {'city': 'Philadelphia', 'state': 'PA', 'zip': '19101'},
            '602': {'city': 'Phoenix', 'state': 'AZ', 'zip': '85001'},
            '619': {'city': 'San Diego', 'state': 'CA', 'zip': '92101'},
            '214': {'city': 'Dallas', 'state': 'TX', 'zip': '75201'},
            '408': {'city': 'San Jose', 'state': 'CA', 'zip': '95101'},
            '512': {'city': 'Austin', 'state': 'TX', 'zip': '78701'},
            '704': {'city': 'Charlotte', 'state': 'NC', 'zip': '28201'},
            '404': {'city': 'Atlanta', 'state': 'GA', 'zip': '30301'},
            '305': {'city': 'Miami', 'state': 'FL', 'zip': '33101'},
            '206': {'city': 'Seattle', 'state': 'WA', 'zip': '98101'},
            '303': {'city': 'Denver', 'state': 'CO', 'zip': '80201'},
            '617': {'city': 'Boston', 'state': 'MA', 'zip': '02101'},
            '702': {'city': 'Las Vegas', 'state': 'NV', 'zip': '89101'},
            '503': {'city': 'Portland', 'state': 'OR', 'zip': '97201'},
            '901': {'city': 'Memphis', 'state': 'TN', 'zip': '38101'},
            '615': {'city': 'Nashville', 'state': 'TN', 'zip': '37201'},
            '813': {'city': 'Tampa', 'state': 'FL', 'zip': '33601'},
            '321': {'city': 'Orlando', 'state': 'FL', 'zip': '32801'},
            '480': {'city': 'Scottsdale', 'state': 'AZ', 'zip': '85251'},
            '916': {'city': 'Sacramento', 'state': 'CA', 'zip': '95814'},
            '253': {'city': 'Tacoma', 'state': 'WA', 'zip': '98401'}
        }
        
        # Initialize new components
        self.investment_calculator = InvestmentScoreCalculator()
        self.visualizer = DataVisualizer()
        self.ml_predictor = MLPredictor()
        
        # 🆕 SESSION DATA STORAGE
        self.session_data = []  # Store analyzed markets in current session
        self.data_file = "remarketintel_session_data.json"  # Persistent storage file
        
        # Load any existing session data
        self.load_session_data()
        
        print(f"✅ Enhanced REMarketIntel initialized")
        print(f"✅ Investment scoring system ready")
        print(f"✅ Data visualization engine ready")
        print(f"✅ ML prediction system ready")
        if self.session_data:
            print(f"📊 Loaded {len(self.session_data)} markets from previous session")
    
    def save_session_data(self):
        """Save current session data to file"""
        try:
            # Convert MarketData objects to dictionaries for JSON serialization
            serializable_data = []
            for market_data in self.session_data:
                data_dict = asdict(market_data)
                # Handle tuple serialization for confidence_interval
                if data_dict.get('confidence_interval') and isinstance(data_dict['confidence_interval'], tuple):
                    data_dict['confidence_interval'] = list(data_dict['confidence_interval'])
                serializable_data.append(data_dict)
            
            with open(self.data_file, 'w') as f:
                json.dump({
                    'markets': serializable_data,
                    'last_updated': datetime.now().isoformat(),
                    'version': '2.0'
                }, f, indent=2)
            
            print(f"💾 Session data saved ({len(self.session_data)} markets)")
        except Exception as e:
            print(f"⚠️  Warning: Could not save session data: {e}")
    
    def load_session_data(self):
        """Load session data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert dictionaries back to MarketData objects
                self.session_data = []
                for market_dict in data.get('markets', []):
                    # Handle confidence_interval conversion back to tuple
                    if market_dict.get('confidence_interval') and isinstance(market_dict['confidence_interval'], list):
                        market_dict['confidence_interval'] = tuple(market_dict['confidence_interval'])
                    
                    market_data = MarketData(**market_dict)
                    self.session_data.append(market_data)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load session data: {e}")
            self.session_data = []
    
    def add_to_session(self, market_data: MarketData):
        """Add market data to current session"""
        # Check if market already exists (update instead of duplicate)
        existing_index = None
        for i, existing_market in enumerate(self.session_data):
            if existing_market.area_code == market_data.area_code:
                existing_index = i
                break
        
        if existing_index is not None:
            self.session_data[existing_index] = market_data
            print(f"🔄 Updated {market_data.city}, {market_data.state} in session data")
        else:
            self.session_data.append(market_data)
            print(f"➕ Added {market_data.city}, {market_data.state} to session data")
        
        # Auto-save session data
        self.save_session_data()
    
    def get_session_summary(self) -> str:
        """Get a summary of current session data"""
        if not self.session_data:
            return "No markets analyzed in current session"
        
        summary = f"📊 Current Session: {len(self.session_data)} markets analyzed\n"
        
        # Show top 5 by investment score
        sorted_markets = sorted([m for m in self.session_data if m.investment_score], 
                              key=lambda x: x.investment_score, reverse=True)[:5]
        
        if sorted_markets:
            summary += "🏆 Top Markets by Investment Score:\n"
            for i, market in enumerate(sorted_markets, 1):
                summary += f"   {i}. {market.city}, {market.state} - {market.investment_score:.1f}\n"
        
        return summary
    
    def clear_session_data(self):
        """Clear all session data"""
        self.session_data = []
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        print("🗑️  Session data cleared")
    
    # Include all original methods (abbreviated for space - you would copy all the original methods here)
    def get_area_info(self, area_code: str) -> Dict:
        """Get area information from area code"""
        if area_code in self.area_code_locations:
            return self.area_code_locations[area_code]
        else:
            print(f"⚠️  Area code {area_code} not found. Using Charlotte, NC as default.")
            return {'city': 'Charlotte', 'state': 'NC', 'zip': '28201'}
    
    def get_fallback_home_value(self, city: str) -> float:
        """Fallback home value estimates by city"""
        value_estimates = {
            'New York': 750000, 'Los Angeles': 850000, 'Chicago': 350000,
            'Houston': 280000, 'Philadelphia': 320000, 'Phoenix': 420000,
            'San Diego': 750000, 'Dallas': 380000, 'San Jose': 1200000,
            'Austin': 450000, 'Charlotte': 380000, 'Atlanta': 350000,
            'Miami': 450000, 'Seattle': 650000, 'Denver': 520000,
            'Boston': 650000, 'Las Vegas': 380000, 'Portland': 520000,
            'Memphis': 180000, 'Nashville': 420000, 'Tampa': 320000,
            'Orlando': 350000, 'Scottsdale': 580000, 'Sacramento': 520000,
            'Tacoma': 450000
        }
        return value_estimates.get(city, 400000)
    
    def estimate_rent_by_city(self, zip_code: str) -> float:
        """Fallback rent estimation by city"""
        rent_estimates = {
            '10001': 3500, '90210': 4000, '60601': 2200, '77001': 1800,
            '19101': 2000, '85001': 1600, '92101': 2800, '75201': 1700,
            '95101': 3200, '78701': 2100, '28201': 1500, '30301': 1800,
            '33101': 2300, '98101': 2600, '80201': 2000, '02101': 2800,
            '89101': 1400, '97201': 2100, '38101': 1200, '37201': 1600,
            '33601': 1800, '32801': 1700, '85251': 2200, '95814': 2100,
            '98401': 1900
        }
        return rent_estimates.get(zip_code, 1800)
    
    def research_market_enhanced(self, area_code: str) -> MarketData:
        """Enhanced market research with investment scoring and ML predictions"""
        
        print(f"\n🏠 Enhanced Market Research for Area Code: {area_code}")
        print("=" * 60)
        
        # Get basic market data (using simplified version of original logic)
        market_data = MarketData(area_code=area_code)
        
        # Get area information
        area_info = self.get_area_info(area_code)
        market_data.city = area_info.get('city', '')
        market_data.state = area_info.get('state', '')
        zip_code = area_info.get('zip', '')
        
        print(f"📍 Analyzing: {market_data.city}, {market_data.state}")
        
        # Get basic market metrics (simplified for demo)
        market_data.average_home_value = self.get_fallback_home_value(market_data.city)
        market_data.rent_estimate = self.estimate_rent_by_city(zip_code)
        
        # Add realistic market data based on city
        regional_data = self.get_regional_market_data(market_data.city)
        for key, value in regional_data.items():
            setattr(market_data, key, value)
        
        # Calculate derived metrics
        if market_data.average_home_value and market_data.median_income:
            current_ratio = market_data.average_home_value / market_data.median_income
            historical_ratio = 6.2
            market_data.overvalued_percentage = ((current_ratio - historical_ratio) / historical_ratio) * 100
        
        if market_data.average_home_value and market_data.rent_estimate:
            annual_rent = market_data.rent_estimate * 12
            market_data.cap_rate = (annual_rent / market_data.average_home_value) * 100
        
        # 🆕 ENHANCED FEATURES START HERE
        
        # 1. Calculate Investment Score
        print("🎯 Calculating investment score...")
        investment_components = self.investment_calculator.calculate_investment_score(market_data)
        market_data.investment_score = investment_components.total_score
        market_data.investment_grade = investment_components.grade
        market_data.risk_level = investment_components.risk_level
        
        print(f"✅ Investment Score: {market_data.investment_score:.1f}/100 (Grade: {market_data.investment_grade})")
        print(f"📊 Risk Level: {market_data.risk_level}")
        
        # 🆕 ADD TO SESSION DATA
        self.add_to_session(market_data)
        
        return market_data, investment_components
    
    def get_regional_market_data(self, city: str) -> Dict:
        """Get comprehensive regional market data"""
        
        # Comprehensive market data by city
        market_data_by_city = {
            'New York': {
                'home_value_growth_yoy': 3.8, 'home_value_growth_5yr': 25.2,
                'for_sale_inventory': 1200, 'price_cut_percentage': 12.3,
                'home_price_forecast': 48, 'median_income': 75000
            },
            'Los Angeles': {
                'home_value_growth_yoy': 5.1, 'home_value_growth_5yr': 35.8,
                'for_sale_inventory': 2800, 'price_cut_percentage': 15.1,
                'home_price_forecast': 45, 'median_income': 70000
            },
            'Chicago': {
                'home_value_growth_yoy': 4.0, 'home_value_growth_5yr': 22.1,
                'for_sale_inventory': 1800, 'price_cut_percentage': 9.8,
                'home_price_forecast': 52, 'median_income': 65000
            },
            'Austin': {
                'home_value_growth_yoy': 5.8, 'home_value_growth_5yr': 42.3,
                'for_sale_inventory': 1400, 'price_cut_percentage': 9.3,
                'home_price_forecast': 57, 'median_income': 68000
            },
            'Charlotte': {
                'home_value_growth_yoy': 5.5, 'home_value_growth_5yr': 35.1,
                'for_sale_inventory': 1100, 'price_cut_percentage': 8.7,
                'home_price_forecast': 61, 'median_income': 62000
            },
            'Miami': {
                'home_value_growth_yoy': 8.2, 'home_value_growth_5yr': 48.6,
                'for_sale_inventory': 2100, 'price_cut_percentage': 14.6,
                'home_price_forecast': 65, 'median_income': 58000
            },
            'Seattle': {
                'home_value_growth_yoy': 2.9, 'home_value_growth_5yr': 32.4,
                'for_sale_inventory': 800, 'price_cut_percentage': 19.4,
                'home_price_forecast': 41, 'median_income': 85000
            },
            'Denver': {
                'home_value_growth_yoy': 4.7, 'home_value_growth_5yr': 38.2,
                'for_sale_inventory': 1300, 'price_cut_percentage': 11.8,
                'home_price_forecast': 53, 'median_income': 70000
            }
        }
        
        # Default values for cities not in the detailed list
        default_data = {
            'home_value_growth_yoy': 4.2, 'home_value_growth_5yr': 28.5,
            'for_sale_inventory': 1000, 'price_cut_percentage': 10.5,
            'home_price_forecast': 52, 'median_income': 60000
        }
        
        return market_data_by_city.get(city, default_data)
    
    def batch_analyze_markets(self, area_codes: List[str]) -> List[MarketData]:
        """Analyze multiple markets and train ML model"""
        
        print(f"\n🔍 Batch Analysis of {len(area_codes)} Markets")
        print("=" * 50)
        
        all_market_data = []
        all_investment_components = []
        
        # Analyze each market
        for i, area_code in enumerate(area_codes, 1):
            print(f"\n📊 Analyzing market {i}/{len(area_codes)}: {area_code}")
            try:
                market_data, investment_components = self.research_market_enhanced(area_code)
                all_market_data.append(market_data)
                all_investment_components.append(investment_components)
                time.sleep(0.5)  # Brief pause to prevent API rate limits
            except Exception as e:
                print(f"❌ Error analyzing {area_code}: {e}")
                continue
        
        if len(all_market_data) < 5:
            print("⚠️  Need at least 5 markets for comprehensive analysis")
            # Still add to session data even if fewer than 5
            return all_market_data
        
        # 2. Train ML Model
        print(f"\n🤖 Training ML prediction model with {len(all_market_data)} markets...")
        training_result = self.ml_predictor.train_model(all_market_data)
        
        if training_result['success']:
            print(f"✅ ML Model trained successfully!")
            print(f"📊 Model Performance - R² Score: {training_result['r2_score']:.3f}")
            print(f"📊 Mean Absolute Error: {training_result['mae']:.2f}%")
            
            # Add ML predictions to each market
            for market_data in all_market_data:
                prediction_result = self.ml_predictor.predict_market_growth(market_data)
                if prediction_result['success']:
                    market_data.predicted_growth_1yr = prediction_result['predicted_growth_1yr']
                    market_data.confidence_interval = prediction_result['confidence_interval']
                    # Update session data with ML predictions
                    self.add_to_session(market_data)
        else:
            print(f"⚠️  ML Training failed: {training_result['message']}")
        
        return all_market_data
    
    def create_comprehensive_report(self, market_data_list: List[MarketData], 
                                  save_visualizations: bool = True) -> None:
        """Create comprehensive investment report with visualizations"""
        
        if not market_data_list:
            print("❌ No market data to analyze")
            return
        
        print(f"\n📊 Creating Comprehensive Investment Report")
        print("=" * 50)
        
        # 1. Display Summary Rankings
        self.display_investment_rankings(market_data_list)
        
        # 2. Create Visualizations
        if save_visualizations:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            print("\n📈 Generating visualizations...")
            
            # Investment Dashboard
            dashboard_path = f"investment_dashboard_{timestamp}.png"
            self.visualizer.create_investment_dashboard(market_data_list, dashboard_path)
            
            # Market Comparison
            comparison_path = f"market_comparison_{timestamp}.png"
            self.visualizer.create_market_comparison_chart(market_data_list, comparison_path)
        
        # 3. ML Model Insights
        if self.ml_predictor.is_trained:
            print("\n🤖 ML Model Insights:")
            feature_importance = self.ml_predictor.get_feature_importance()
            
            print("📊 Most Important Prediction Factors:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
                print(f"   {i}. {feature.replace('_', ' ').title()}: {importance:.3f}")
        
        # 4. Investment Recommendations
        self.generate_investment_recommendations(market_data_list)
    
    def display_investment_rankings(self, market_data_list: List[MarketData]) -> None:
        """Display ranked investment opportunities"""
        
        # Sort by investment score
        sorted_markets = sorted(market_data_list, key=lambda x: x.investment_score or 0, reverse=True)
        
        print(f"\n🏆 TOP INVESTMENT OPPORTUNITIES")
        print("=" * 80)
        print(f"{'Rank':<4} {'Market':<20} {'Score':<6} {'Grade':<5} {'Risk':<12} {'Cap Rate':<8} {'YoY Growth':<10}")
        print("-" * 80)
        
        for i, market in enumerate(sorted_markets[:10], 1):
            location = f"{market.city}, {market.state}"
            score = f"{market.investment_score:.1f}" if market.investment_score else "N/A"
            grade = market.investment_grade or "N/A"
            risk = market.risk_level or "N/A"
            cap_rate = f"{market.cap_rate:.2f}%" if market.cap_rate else "N/A"
            yoy_growth = f"{market.home_value_growth_yoy:.1f}%" if market.home_value_growth_yoy else "N/A"
            
            print(f"{i:<4} {location:<20} {score:<6} {grade:<5} {risk:<12} {cap_rate:<8} {yoy_growth:<10}")
    
    def generate_investment_recommendations(self, market_data_list: List[MarketData]) -> None:
        """Generate personalized investment recommendations"""
        
        print(f"\n💡 INVESTMENT RECOMMENDATIONS")
        print("=" * 50)
        
        sorted_markets = sorted(market_data_list, key=lambda x: x.investment_score or 0, reverse=True)
        
        # Best overall opportunities
        top_3 = sorted_markets[:3]
        print(f"\n🥇 TOP OVERALL OPPORTUNITIES:")
        for i, market in enumerate(top_3, 1):
            print(f"   {i}. {market.city}, {market.state}")
            print(f"      💰 Investment Score: {market.investment_score:.1f} (Grade: {market.investment_grade})")
            print(f"      📈 Expected Growth: {market.home_value_growth_yoy:.1f}% YoY")
            if market.predicted_growth_1yr:
                print(f"      🤖 ML Prediction: {market.predicted_growth_1yr:.1f}% (Next 12 months)")
            print(f"      🏠 Cap Rate: {market.cap_rate:.2f}%")
        
        # Best cash flow markets
        cash_flow_markets = sorted([m for m in market_data_list if m.cap_rate], 
                                 key=lambda x: x.cap_rate, reverse=True)[:3]
        print(f"\n💸 BEST CASH FLOW OPPORTUNITIES:")
        for i, market in enumerate(cash_flow_markets, 1):
            print(f"   {i}. {market.city}, {market.state} - {market.cap_rate:.2f}% cap rate")
        
        # Best growth markets
        growth_markets = sorted([m for m in market_data_list if m.home_value_growth_yoy], 
                              key=lambda x: x.home_value_growth_yoy, reverse=True)[:3]
        print(f"\n📈 BEST APPRECIATION OPPORTUNITIES:")
        for i, market in enumerate(growth_markets, 1):
            print(f"   {i}. {market.city}, {market.state} - {market.home_value_growth_yoy:.1f}% YoY growth")
        
        # Risk warnings
        high_risk_markets = [m for m in market_data_list if m.risk_level in ['High', 'Very High']]
        if high_risk_markets:
            print(f"\n⚠️  HIGH RISK MARKETS TO WATCH:")
            for market in high_risk_markets[:3]:
                print(f"   • {market.city}, {market.state} - {market.risk_level} risk")

def main_enhanced():
    """Enhanced main function with new features"""
    
    print("🏠 REMarketIntel ENHANCED - Investment Analysis Platform")
    print("Powered by ML Predictions, Investment Scoring & Advanced Visualization")
    print("=" * 80)
    
    # Check environment
    if not os.path.exists('.env'):
        print("❌ .env file not found! Please create one with your API keys.")
        return
    
    try:
        # Initialize enhanced researcher
        researcher = EnhancedRealEstateMarketResearcher()
        
        while True:
            print(f"\n🎯 ENHANCED ANALYSIS OPTIONS:")
            print("1. Single Market Analysis (Enhanced)")
            print("2. Multi-Market Investment Analysis (NEW)")
            print("3. Create Investment Dashboard (NEW)")
            print("4. ML Market Predictions (NEW)")
            print("5. Quick Demo with Sample Markets")
            print("6. View Session Summary")
            print("7. Clear Session Data")
            print("8. Exit")
            
            choice = input("\n📊 Select option (1-8): ").strip()
            
            if choice == '1':
                # Single market analysis
                area_code = input("📱 Enter area code: ").strip()
                if len(area_code) == 3 and area_code.isdigit():
                    market_data, components = researcher.research_market_enhanced(area_code)
                    researcher.display_enhanced_results(market_data, components)
                else:
                    print("❌ Invalid area code")
            
            elif choice == '2':
                # Multi-market analysis
                print("📍 Enter area codes separated by commas (e.g., 212,310,312):")
                area_codes_input = input("Area codes: ").strip()
                area_codes = [code.strip() for code in area_codes_input.split(',')]
                
                if all(len(code) == 3 and code.isdigit() for code in area_codes):
                    markets = researcher.batch_analyze_markets(area_codes)
                    researcher.create_comprehensive_report(markets)
                else:
                    print("❌ Invalid area codes")
            
            elif choice == '3':
                # Create dashboard for session data
                if not researcher.session_data:
                    print("❌ No analysis data available. Please run option 1, 2, 4, or 5 first.")
                    print("\n💡 Tip: Analyze some markets first, then return to create visualizations!")
                else:
                    print(f"📊 Creating dashboard with {len(researcher.session_data)} markets from session...")
                    researcher.create_comprehensive_report(researcher.session_data, save_visualizations=True)
            
            elif choice == '4':
                # ML predictions demo
                sample_codes = ['212', '310', '312', '704', '512']  # NYC, LA, Chicago, Charlotte, Austin
                print(f"🤖 Running ML predictions on sample markets: {', '.join(sample_codes)}")
                markets = researcher.batch_analyze_markets(sample_codes)
                researcher.create_comprehensive_report(markets)
            
            elif choice == '5':
                # Quick demo
                demo_codes = ['704', '512', '404', '305']  # Charlotte, Austin, Atlanta, Miami
                print(f"🚀 Quick demo with: Charlotte, Austin, Atlanta, Miami")
                markets = researcher.batch_analyze_markets(demo_codes)
                researcher.create_comprehensive_report(markets)
            
            elif choice == '6':
                # View session summary
                print(f"\n{researcher.get_session_summary()}")
            
            elif choice == '7':
                # Clear session data
                confirm = input("⚠️  Are you sure you want to clear all session data? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    researcher.clear_session_data()
                else:
                    print("❌ Operation cancelled")
            
            elif choice == '8':
                break
            
            else:
                print("❌ Invalid option")
            
            
            # Continue prompt
            if choice in ['1', '2', '3', '4', '5']:
                continue_analysis = input("\n🔄 Continue analysis? (y/n): ").strip().lower()
                if continue_analysis not in ['y', 'yes']:
                    break
        
        print("\n🎉 Thank you for using REMarketIntel Enhanced!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def run_interactive_demo():
    """Run an interactive demo of the enhanced features"""
    
    print("\n🚀 REMarketIntel Enhanced - Interactive Demo")
    print("=" * 60)
    
    # Sample data for demo
    demo_area_codes = ['704', '512', '404', '305', '206']  # Charlotte, Austin, Atlanta, Miami, Seattle
    
    try:
        researcher = EnhancedRealEstateMarketResearcher()
        
        print("📊 Analyzing 5 sample markets for demonstration...")
        print("Markets: Charlotte, Austin, Atlanta, Miami, Seattle")
        
        # Run batch analysis
        markets = researcher.batch_analyze_markets(demo_area_codes)
        
        if len(markets) >= 3:
            print(f"\n✅ Analysis complete! Generating comprehensive report...")
            researcher.create_comprehensive_report(markets, save_visualizations=True)
            
            # Save results
            csv_file = save_enhanced_results_to_csv(markets)
            pdf_file = generate_pdf_report(markets)
            
            print(f"\n📁 FILES GENERATED:")
            if csv_file:
                print(f"   📊 Data: {csv_file}")
            if pdf_file:
                print(f"   📄 Report: {pdf_file}")
            
            print(f"   📈 Charts: investment_dashboard_*.png")
            print(f"   📊 Comparison: market_comparison_*.png")
            
            # Show session summary
            print(f"\n{researcher.get_session_summary()}")
        else:
            print("❌ Demo failed - insufficient data")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")

# Add missing method to handle matplotlib backend issues
def setup_matplotlib():
    """Setup matplotlib with appropriate backend"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for compatibility
        import matplotlib.pyplot as plt
        plt.style.use('default')  # Use default style if seaborn-v0_8 not available
        return True
    except Exception as e:
        print(f"⚠️  Matplotlib setup warning: {e}")
        return False

# Update the DataVisualizer class to handle backend issues
class DataVisualizer:
    """Enhanced data visualization for market analysis"""
    
    def __init__(self):
        # Set style for better-looking plots with fallback
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        except:
            try:
                plt.style.use('ggplot')  # Fallback style
            except:
                plt.style.use('default')  # Final fallback
        
    def create_investment_dashboard(self, market_data_list: List[MarketData], 
                                  save_path: str = None) -> None:
        """Create comprehensive investment dashboard"""
        
        if not setup_matplotlib():
            print("❌ Could not setup matplotlib for visualization")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Prepare data
            df = pd.DataFrame([asdict(md) for md in market_data_list])
            df['location'] = df['city'] + ', ' + df['state']
            
            # 1. Investment Scores Ranking (Top Left)
            ax1 = fig.add_subplot(gs[0, :2])
            if 'investment_score' in df.columns and not df['investment_score'].isna().all():
                top_markets = df.nlargest(min(10, len(df)), 'investment_score')
                bars = ax1.barh(range(len(top_markets)), top_markets['investment_score'])
                ax1.set_yticks(range(len(top_markets)))
                ax1.set_yticklabels(top_markets['location'])
                ax1.set_xlabel('Investment Score')
                ax1.set_title('Top Investment Markets by Score', fontsize=14, fontweight='bold')
                
                # Color bars by grade
                for i, (_, row) in enumerate(top_markets.iterrows()):
                    score = row.get('investment_score', 0)
                    if score >= 80:
                        bars[i].set_color('green')
                    elif score >= 70:
                        bars[i].set_color('orange')
                    else:
                        bars[i].set_color('red')
            else:
                ax1.text(0.5, 0.5, 'No Investment Score Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Investment Scores', fontsize=14)
            
            # 2. Cap Rate vs Home Value (Top Right)
            ax2 = fig.add_subplot(gs[0, 2:])
            if ('cap_rate' in df.columns and 'average_home_value' in df.columns and 
                not df['cap_rate'].isna().all() and not df['average_home_value'].isna().all()):
                
                # Filter out NaN values
                plot_df = df.dropna(subset=['cap_rate', 'average_home_value', 'investment_score'])
                if len(plot_df) > 0:
                    scatter = ax2.scatter(plot_df['average_home_value']/1000, plot_df['cap_rate'], 
                                        c=plot_df['investment_score'], cmap='RdYlGn', s=100, alpha=0.7)
                    ax2.set_xlabel('Average Home Value ($000s)')
                    ax2.set_ylabel('Cap Rate (%)')
                    ax2.set_title('Cap Rate vs Home Value', fontsize=14, fontweight='bold')
                    try:
                        plt.colorbar(scatter, ax=ax2, label='Investment Score')
                    except:
                        pass  # Skip colorbar if it fails
                else:
                    ax2.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No Cap Rate/Value Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Cap Rate vs Home Value', fontsize=14)
            
            # 3. Growth Trends (Middle Left)
            ax3 = fig.add_subplot(gs[1, :2])
            if 'home_value_growth_yoy' in df.columns and not df['home_value_growth_yoy'].isna().all():
                growth_df = df.dropna(subset=['home_value_growth_yoy']).sort_values('home_value_growth_yoy', ascending=True)
                if len(growth_df) > 0:
                    bars = ax3.barh(range(len(growth_df)), growth_df['home_value_growth_yoy'])
                    ax3.set_yticks(range(len(growth_df)))
                    ax3.set_yticklabels(growth_df['location'])
                    ax3.set_xlabel('YoY Growth (%)')
                    
                    # Color by growth rate
                    for i, growth in enumerate(growth_df['home_value_growth_yoy']):
                        if growth >= 6:
                            bars[i].set_color('darkgreen')
                        elif growth >= 3:
                            bars[i].set_color('lightgreen')
                        elif growth >= 0:
                            bars[i].set_color('yellow')
                        else:
                            bars[i].set_color('red')
                else:
                    ax3.text(0.5, 0.5, 'No Growth Data', ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No Growth Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Home Value Growth (Year-over-Year)', fontsize=14, fontweight='bold')
            
            # 4. Summary Statistics (Middle Right)
            ax4 = fig.add_subplot(gs[1, 2:])
            summary_text = f"📊 PORTFOLIO SUMMARY\n\n"
            summary_text += f"Markets Analyzed: {len(df)}\n"
            
            if 'investment_score' in df.columns:
                avg_score = df['investment_score'].mean()
                summary_text += f"Avg Investment Score: {avg_score:.1f}\n"
            
            if 'cap_rate' in df.columns:
                avg_cap_rate = df['cap_rate'].mean()
                summary_text += f"Avg Cap Rate: {avg_cap_rate:.2f}%\n"
            
            if 'home_value_growth_yoy' in df.columns:
                avg_growth = df['home_value_growth_yoy'].mean()
                summary_text += f"Avg YoY Growth: {avg_growth:.1f}%\n"
            
            # Find best markets
            if 'investment_score' in df.columns and not df['investment_score'].isna().all():
                best_market = df.loc[df['investment_score'].idxmax()]
                summary_text += f"\n🏆 TOP MARKET:\n{best_market['location']}\n"
                summary_text += f"Score: {best_market['investment_score']:.1f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            # 5. Risk Analysis (Bottom Left)
            ax5 = fig.add_subplot(gs[2, :2])
            if 'risk_level' in df.columns:
                risk_counts = df['risk_level'].value_counts()
                if len(risk_counts) > 0:
                    colors = {'Low': 'green', 'Low-Medium': 'lightgreen', 'Medium': 'yellow', 
                             'Medium-High': 'orange', 'High': 'red', 'Very High': 'darkred'}
                    bar_colors = [colors.get(risk, 'gray') for risk in risk_counts.index]
                    ax5.bar(risk_counts.index, risk_counts.values, color=bar_colors)
                    ax5.set_ylabel('Number of Markets')
                    ax5.tick_params(axis='x', rotation=45)
                else:
                    ax5.text(0.5, 0.5, 'No Risk Data', ha='center', va='center', transform=ax5.transAxes)
            else:
                ax5.text(0.5, 0.5, 'No Risk Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Risk Distribution', fontsize=14, fontweight='bold')
            
            # 6. Investment Grades (Bottom Right)
            ax6 = fig.add_subplot(gs[2, 2:])
            if 'investment_grade' in df.columns:
                grade_counts = df['investment_grade'].value_counts()
                if len(grade_counts) > 0:
                    ax6.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%')
                else:
                    ax6.text(0.5, 0.5, 'No Grade Data', ha='center', va='center', transform=ax6.transAxes)
            else:
                ax6.text(0.5, 0.5, 'No Grade Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Investment Grade Distribution', fontsize=14, fontweight='bold')
            
            # 7. ML Predictions (Bottom)
            ax7 = fig.add_subplot(gs[3, :])
            if 'predicted_growth_1yr' in df.columns and not df['predicted_growth_1yr'].isna().all():
                pred_df = df.dropna(subset=['predicted_growth_1yr'])
                if len(pred_df) > 0:
                    bars = ax7.bar(pred_df['location'], pred_df['predicted_growth_1yr'])
                    ax7.set_ylabel('Predicted Growth (%)')
                    ax7.set_title('ML Predicted 1-Year Growth', fontsize=14, fontweight='bold')
                    ax7.tick_params(axis='x', rotation=45)
                    
                    # Color bars by prediction
                    for i, pred in enumerate(pred_df['predicted_growth_1yr']):
                        if pred >= 5:
                            bars[i].set_color('green')
                        elif pred >= 2:
                            bars[i].set_color('lightgreen')
                        elif pred >= 0:
                            bars[i].set_color('yellow')
                        else:
                            bars[i].set_color('red')
                else:
                    ax7.text(0.5, 0.5, 'No ML Prediction Data', ha='center', va='center', transform=ax7.transAxes)
            else:
                ax7.text(0.5, 0.5, 'No ML Prediction Data Available', ha='center', va='center', 
                        transform=ax7.transAxes, fontsize=12)
            ax7.set_title('ML Predictions', fontsize=14, fontweight='bold')
            
            # Main title
            fig.suptitle('🏠 REMarketIntel Investment Dashboard', fontsize=20, fontweight='bold', y=0.98)
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"📊 Dashboard saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            plt.close('all')
    
    def create_market_comparison_chart(self, market_data_list: List[MarketData], 
                                    save_path: str = None) -> None:
        """Create side-by-side market comparison"""
        
        if len(market_data_list) < 2:
            print("⚠️  Need at least 2 markets for comparison")
            return
        
        if not setup_matplotlib():
            print("❌ Could not setup matplotlib for visualization")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            # Prepare data
            df = pd.DataFrame([asdict(md) for md in market_data_list])
            df['location'] = df['city'] + ', ' + df['state']
            
            # Create comparison figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Investment Scores
            if 'investment_score' in df.columns and not df['investment_score'].isna().all():
                score_df = df.dropna(subset=['investment_score'])
                bars1 = ax1.bar(range(len(score_df)), score_df['investment_score'])
                ax1.set_xticks(range(len(score_df)))
                ax1.set_xticklabels(score_df['location'], rotation=45, ha='right')
                ax1.set_ylabel('Investment Score')
                
                # Color bars by score
                for bar, score in zip(bars1, score_df['investment_score']):
                    if score >= 80:
                        bar.set_color('green')
                    elif score >= 70:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            else:
                ax1.text(0.5, 0.5, 'No Investment Score Data', ha='center', va='center', transform=ax1.transAxes)
            
            ax1.set_title('Investment Scores Comparison', fontsize=14, fontweight='bold')
            
            # 2. Key Metrics Comparison
            metrics = ['cap_rate', 'home_value_growth_yoy', 'price_cut_percentage']
            available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
            
            if available_metrics:
                x = np.arange(len(df))
                width = 0.25
                
                for i, metric in enumerate(available_metrics[:3]):  # Max 3 metrics to fit
                    metric_df = df.dropna(subset=[metric])
                    if len(metric_df) > 0:
                        ax2.bar(x + i*width, df[metric].fillna(0), width, 
                            label=metric.replace('_', ' ').title())
                
                ax2.set_xlabel('Markets')
                ax2.set_xticks(x + width)
                ax2.set_xticklabels(df['location'], rotation=45, ha='right')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No Metrics Data Available', ha='center', va='center', transform=ax2.transAxes)
            
            ax2.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
            
            # 3. Market Summary Table
            ax3.axis('tight')
            ax3.axis('off')
            
            # Create summary table
            table_data = []
            headers = ['Market', 'Score', 'Grade', 'Cap Rate', 'Growth']
            table_data.append(headers)
            
            for _, row in df.iterrows():
                table_row = [
                    f"{row['city']}, {row['state']}",
                    f"{row.get('investment_score', 0):.1f}" if pd.notna(row.get('investment_score')) else "N/A",
                    row.get('investment_grade', 'N/A'),
                    f"{row.get('cap_rate', 0):.2f}%" if pd.notna(row.get('cap_rate')) else "N/A",
                    f"{row.get('home_value_growth_yoy', 0):.1f}%" if pd.notna(row.get('home_value_growth_yoy')) else "N/A"
                ]
                table_data.append(table_row)
            
            table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            ax3.set_title('Market Summary', fontsize=14, fontweight='bold')
            
            # 4. Risk vs Return
            if ('investment_score' in df.columns and 'cap_rate' in df.columns and 
                not df['investment_score'].isna().all() and not df['cap_rate'].isna().all()):
                
                plot_df = df.dropna(subset=['investment_score', 'cap_rate'])
                if len(plot_df) > 0:
                    # Risk score is inverse of investment score
                    risk_scores = 100 - plot_df['investment_score']
                    scatter = ax4.scatter(risk_scores, plot_df['cap_rate'], s=200, alpha=0.6)
                    
                    # Add market labels
                    for i, (_, row) in enumerate(plot_df.iterrows()):
                        ax4.annotate(row['city'], 
                                (100 - row['investment_score'], row['cap_rate']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    ax4.set_xlabel('Risk Score (Higher = More Risk)')
                    ax4.set_ylabel('Cap Rate (%)')
                    
                    # Add quadrant lines
                    ax4.axhline(y=plot_df['cap_rate'].median(), color='gray', linestyle='--', alpha=0.5)
                    ax4.axvline(x=risk_scores.median(), color='gray', linestyle='--', alpha=0.5)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'No Risk/Return Data', ha='center', va='center', transform=ax4.transAxes)
            
            ax4.set_title('Risk vs Return Matrix', fontsize=14, fontweight='bold')
            
            plt.suptitle('Market Comparison Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"📊 Comparison chart saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating comparison chart: {e}")
            plt.close('all')

# Add method to display enhanced results
def display_enhanced_results(self, market_data: MarketData, components: InvestmentScoreComponents):
    """Display enhanced results with investment scoring"""
    
    print(f"\n🏡 ENHANCED MARKET ANALYSIS RESULTS")
    print(f"📍 Location: {market_data.city}, {market_data.state}")
    print("=" * 70)
    
    # Investment Score Section
    print(f"\n🎯 INVESTMENT ANALYSIS:")
    print(f"   Overall Score: {components.total_score:.1f}/100")
    print(f"   Investment Grade: {components.grade}")
    print(f"   Risk Level: {components.risk_level}")
    
    print(f"\n📊 SCORE BREAKDOWN:")
    print(f"   💰 Cash Flow Score: {components.cash_flow_score:.1f}/100")
    print(f"   📈 Appreciation Score: {components.appreciation_score:.1f}/100")
    print(f"   🏛️  Market Stability: {components.market_stability_score:.1f}/100")
    print(f"   💵 Affordability Score: {components.affordability_score:.1f}/100")
    print(f"   🔄 Liquidity Score: {components.liquidity_score:.1f}/100")
    
    # Traditional metrics
    print(f"\n📈 MARKET FUNDAMENTALS:")
    print(f"   🏠 Average Home Value: ${market_data.average_home_value:,.0f}")
    print(f"   📊 YoY Growth: {market_data.home_value_growth_yoy:.1f}%")
    print(f"   💰 Cap Rate: {market_data.cap_rate:.2f}%")
    print(f"   📦 Inventory: {market_data.for_sale_inventory:,} units")
    
    # ML Predictions
    if market_data.predicted_growth_1yr:
        print(f"\n🤖 ML PREDICTIONS:")
        print(f"   📈 Predicted 1-Year Growth: {market_data.predicted_growth_1yr:.1f}%")
        if market_data.confidence_interval:
            ci_lower, ci_upper = market_data.confidence_interval
            print(f"   📊 Confidence Interval: {ci_lower:.1f}% to {ci_upper:.1f}%")
    
    # Investment Recommendation
    print(f"\n💡 INVESTMENT RECOMMENDATION:")
    if components.total_score >= 80:
        print("   🟢 STRONG BUY - Excellent investment opportunity")
    elif components.total_score >= 70:
        print("   🟡 BUY - Good investment with solid fundamentals")
    elif components.total_score >= 60:
        print("   🟡 HOLD/CONSIDER - Mixed signals, analyze further")
    else:
        print("   🔴 AVOID - High risk, poor fundamentals")

# Add the display method to the class
EnhancedRealEstateMarketResearcher.display_enhanced_results = display_enhanced_results

def check_enhanced_dependencies():
    """Check and install enhanced dependencies"""
    import importlib
    
    # Package mapping: pip name -> import name
    required_packages = {
        'requests': 'requests',
        'pandas': 'pandas', 
        'python-dotenv': 'dotenv',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scikit-learn': 'sklearn'
    }
    
    missing_packages = []
    
    for pip_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        import subprocess
        import sys
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        print("🔄 Please restart the script to use new features")
        return False
    
    print("✅ All dependencies are installed")
    return True

def create_sample_env_enhanced():
    """Create enhanced sample .env file"""
    sample_content = """# REMarketIntel Enhanced - API Configuration
# Required: RentCast API for real estate data
RENTCAST_API_KEY=your_rentcast_api_key_here

# Optional: FRED API for economic data (enhances analysis)
FRED_API_KEY=your_fred_api_key_here

# Get RentCast API key: https://www.rentcast.io/api
# Get FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

# Example:
# RENTCAST_API_KEY=abc123def456ghi789
# FRED_API_KEY=xyz789uvw456rst123
"""
    
    try:
        if not os.path.exists('.env.sample'):
            with open('.env.sample', 'w') as f:
                f.write(sample_content)
            print("📝 Created .env.sample file")
    except Exception as e:
        print(f"❌ Error creating sample file: {e}")

def save_enhanced_results_to_csv(market_data_list: List[MarketData], filename: str = None):
    """Save enhanced results with investment scores to CSV"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"REMarketIntel_Enhanced_Results_{timestamp}.csv"
    
    data_dicts = []
    for market_data in market_data_list:
        data_dict = {
            'Area_Code': market_data.area_code,
            'City': market_data.city,
            'State': market_data.state,
            'Average_Home_Value': market_data.average_home_value,
            'YoY_Growth_Percent': market_data.home_value_growth_yoy,
            'Five_Year_Growth_Percent': market_data.home_value_growth_5yr,
            'For_Sale_Inventory': market_data.for_sale_inventory,
            'Price_Cut_Percent': market_data.price_cut_percentage,
            'Price_Forecast_Score': market_data.home_price_forecast,
            'Overvalued_Percent': market_data.overvalued_percentage,
            'Cap_Rate_Percent': market_data.cap_rate,
            'Median_Income': market_data.median_income,
            'Rent_Estimate': market_data.rent_estimate,
            # Enhanced fields
            'Investment_Score': market_data.investment_score,
            'Investment_Grade': market_data.investment_grade,
            'Risk_Level': market_data.risk_level,
            'Predicted_Growth_1yr': market_data.predicted_growth_1yr,
            'Research_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        data_dicts.append(data_dict)
    
    try:
        df = pd.DataFrame(data_dicts)
        df.to_csv(filename, index=False)
        print(f"💾 Enhanced results saved to {filename}")
        print(f"📊 Exported {len(data_dicts)} market analysis records")
        return filename
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return None

def generate_pdf_report(market_data_list: List[MarketData], filename: str = None):
    """Generate a comprehensive PDF investment report"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"REMarketIntel_Investment_Report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("REMarketIntel Investment Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Executive Summary
        summary_text = f"""
        <b>Executive Summary</b><br/>
        This report analyzes {len(market_data_list)} real estate markets using advanced investment scoring,
        machine learning predictions, and comprehensive market fundamentals analysis.
        Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        summary = Paragraph(summary_text, styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 12))
        
        # Top Markets Table
        if market_data_list:
            sorted_markets = sorted(market_data_list, key=lambda x: x.investment_score or 0, reverse=True)
            
            # Create table data
            table_data = [['Rank', 'Market', 'Investment Score', 'Grade', 'Cap Rate', 'YoY Growth']]
            
            for i, market in enumerate(sorted_markets[:10], 1):
                row = [
                    str(i),
                    f"{market.city}, {market.state}",
                    f"{market.investment_score:.1f}" if market.investment_score else "N/A",
                    market.investment_grade or "N/A",
                    f"{market.cap_rate:.2f}%" if market.cap_rate else "N/A",
                    f"{market.home_value_growth_yoy:.1f}%" if market.home_value_growth_yoy else "N/A"
                ]
                table_data.append(row)
            
            # Create and style table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        # Build PDF
        doc.build(story)
        print(f"📄 PDF report generated: {filename}")
        return filename
        
    except ImportError:
        print("⚠️  ReportLab not installed. Install with: pip install reportlab")
        return None
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return None

def run_interactive_demo():
    """Run an interactive demo of the enhanced features"""
    
    print("\n🚀 REMarketIntel Enhanced - Interactive Demo")
    print("=" * 60)
    
    # Sample data for demo
    demo_area_codes = ['704', '512', '404', '305', '206']  # Charlotte, Austin, Atlanta, Miami, Seattle
    
    try:
        researcher = EnhancedRealEstateMarketResearcher()
        
        print("📊 Analyzing 5 sample markets for demonstration...")
        print("Markets: Charlotte, Austin, Atlanta, Miami, Seattle")
        
        # Run batch analysis
        markets = researcher.batch_analyze_markets(demo_area_codes)
        
        if len(markets) >= 3:
            print(f"\n✅ Analysis complete! Generating comprehensive report...")
            researcher.create_comprehensive_report(markets, save_visualizations=True)
            
            # Save results
            csv_file = save_enhanced_results_to_csv(markets)
            pdf_file = generate_pdf_report(markets)
            
            print(f"\n📁 FILES GENERATED:")
            if csv_file:
                print(f"   📊 Data: {csv_file}")
            if pdf_file:
                print(f"   📄 Report: {pdf_file}")
            
            print(f"   📈 Charts: investment_dashboard_*.png")
            print(f"   📊 Comparison: market_comparison_*.png")
        else:
            print("❌ Demo failed - insufficient data")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    print("🏠 REMarketIntel Enhanced - Loading...")
    
    # Check dependencies first
    if not check_enhanced_dependencies():
        print("\n❌ Dependency installation completed. Please run the script again.")
        input("Press Enter to exit...")
        exit(1)
    
    # Import required modules after dependency check
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please restart the script.")
        input("Press Enter to exit...")
        exit(1)
    
    # Create sample .env if needed
    if not os.path.exists('.env'):
        create_sample_env_enhanced()
        print("❌ Please create .env file with your API keys before continuing")
        print("📝 A sample .env file has been created for reference")
        input("Press Enter to exit...")
        exit(1)
    
    # Check if this is a quick demo run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_interactive_demo()
    else:
        main_enhanced()