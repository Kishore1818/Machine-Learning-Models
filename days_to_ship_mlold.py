import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import joblib

# Set random seed for reproducibility
np.random.seed(42)

class DaysToShipPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.results = {}
        
    def load_and_preprocess_data(self, df):
        """
        Load and preprocess the data
        """
        print("Starting data preprocessing...")
        
        # Create target variable: days to ship
        df['SalesOrderDate'] = pd.to_datetime(df['SalesOrderDate'], errors='coerce')
        df['ActualShipDate'] = pd.to_datetime(df['ActualShipDate'], errors='coerce')
        df['days_to_ship'] = (df['ActualShipDate'] - df['SalesOrderDate']).dt.days
        
        # Remove rows where we can't calculate days to ship
        df = df.dropna(subset=['days_to_ship'])
        
        # Remove negative days (data quality issues)
        df = df[df['days_to_ship'] >= 0]
        
        print(f"Dataset shape after preprocessing: {df.shape}")
        print(f"Days to ship - Min: {df['days_to_ship'].min()}, Max: {df['days_to_ship'].max()}, Mean: {df['days_to_ship'].mean():.2f}")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create features for ML models
        """
        print("Starting feature engineering...")
        
        # Date features
        if 'SalesOrderDate' in df.columns:
            df['order_month'] = df['SalesOrderDate'].dt.month
            df['order_day_of_week'] = df['SalesOrderDate'].dt.dayofweek
            df['order_quarter'] = df['SalesOrderDate'].dt.quarter
        
        # Backorder features
        if 'WasTheOrderEverBackordered' in df.columns:
            df['was_backordered'] = df['WasTheOrderEverBackordered'].fillna(0)
        else:
            df['was_backordered'] = 0
            
        if 'DateBackordered' in df.columns:
            df['DateBackordered'] = pd.to_datetime(df['DateBackordered'], errors='coerce')
            if 'SalesOrderDate' in df.columns:
                df['backorder_delay'] = (df['DateBackordered'] - df['SalesOrderDate']).dt.days
                df['backorder_delay'] = df['backorder_delay'].fillna(0)
            else:
                df['backorder_delay'] = 0
        else:
            df['backorder_delay'] = 0
        
        # Quantity features
        if 'ShipmentQuantity' in df.columns and 'OriginalQuantityOrdered' in df.columns:
            df['quantity_ratio'] = df['ShipmentQuantity'] / (df['OriginalQuantityOrdered'] + 1)
        else:
            df['quantity_ratio'] = 1.0
            
        if 'QuantityOpen' in df.columns and 'OriginalQuantityOrdered' in df.columns:
            df['quantity_open_ratio'] = df['QuantityOpen'] / (df['OriginalQuantityOrdered'] + 1)
        else:
            df['quantity_open_ratio'] = 0.0
        
        # Product features
        if all(col in df.columns for col in ['SkuLength', 'SkuWidth', 'SkuHeight']):
            df['product_volume'] = df['SkuLength'] * df['SkuWidth'] * df['SkuHeight']
            if 'SkuWeight' in df.columns:
                df['weight_to_volume_ratio'] = df['SkuWeight'] / (df['product_volume'] + 1)
            else:
                df['weight_to_volume_ratio'] = 0.0
        else:
            df['product_volume'] = 1.0
            df['weight_to_volume_ratio'] = 0.0
        
        # Inventory features
        if 'ShipmentQuantity' in df.columns and 'BalanceOnHand' in df.columns:
            df['inventory_turnover'] = df['ShipmentQuantity'] / (df['BalanceOnHand'] + 1)
        else:
            df['inventory_turnover'] = 0.0
            
        if 'QuantityCommitted' in df.columns and 'BalanceOnHand' in df.columns:
            df['commitment_ratio'] = df['QuantityCommitted'] / (df['BalanceOnHand'] + 1)
        else:
            df['commitment_ratio'] = 0.0
        
        print("Feature engineering completed.")
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for ML models
        """
        print("Preparing features for ML models...")
        
        # Define categorical and numerical features
        categorical_features = [
            'CompanyCode', 'CustomerType', 'AutoPrintFlag', 'GovtEndUserType',
            'PriorityCode', 'VendorClass', 'VendorCode', 'MediaCode',
            'CategorySubcategory', 'ProductClass', 'ProductGroup',
            'ShipFromBranch', 'ShipToState', 'ShipToCountry', 'CarrierCode'
        ]
        
        numerical_features = [
            'BranchCustomerNumber', 'OriginalQuantityOrdered', 'QuantityOpen',
            'ShipmentQuantity', 'SkuLength', 'SkuWidth', 'SkuHeight', 'SkuWeight',
            'UnitCost', 'BalanceOnHand', 'QuantityOnOrder', 'QuantityCommitted',
            'order_month', 'order_day_of_week', 'order_quarter', 'was_backordered',
            'backorder_delay', 'quantity_ratio', 'quantity_open_ratio',
            'product_volume', 'weight_to_volume_ratio', 'inventory_turnover',
            'commitment_ratio'
        ]
        
        # Handle missing values
        for col in numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Encode categorical features
        X_encoded = df[numerical_features].copy()
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                X_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # Target variable
        y = df['days_to_ship']
        
        print(f"Final feature matrix shape: {X_encoded.shape}")
        return X_encoded, y
    
    def train_models(self, X, y):
        """
        Train multiple ML models
        """
        print("Training multiple ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, eval_metric='rmse'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # Train and evaluate models
        results = []
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for models that need it
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'SVR']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # Train model
            model.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'SVR']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2_Score': r2,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std()
            })
            
            # Store model
            self.models[name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = np.abs(model.coef_)
        
        self.results = pd.DataFrame(results)
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        return self.results
    
    def get_model_comparison(self):
        """
        Display model comparison table
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        # Sort by R2 score
        comparison = self.results.sort_values('R2_Score', ascending=False)
        
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2_Score':<12} {'CV_R2_Mean':<12} {'CV_R2_Std':<12}")
        print("-" * 80)
        
        for _, row in comparison.iterrows():
            print(f"{row['Model']:<20} {row['RMSE']:<10.3f} {row['MAE']:<10.3f} {row['R2_Score']:<12.3f} {row['CV_R2_Mean']:<12.3f} {row['CV_R2_Std']:<12.3f}")
        
        # Select best model
        best_model_name = comparison.iloc[0]['Model']
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   R2 Score: {comparison.iloc[0]['R2_Score']:.3f}")
        print(f"   RMSE: {comparison.iloc[0]['RMSE']:.3f} days")
        print(f"   MAE: {comparison.iloc[0]['MAE']:.3f} days")
        
        return best_model_name, comparison
    
    def save_best_model(self, best_model_name):
        """
        Save the best model and preprocessing components
        """
        print(f"\nSaving best model: {best_model_name}")
        
        # Save model
        best_model = self.models[best_model_name]
        joblib.dump(best_model, f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
        
        # Save scalers and encoders
        joblib.dump(self.scalers, 'scalers.pkl')
        joblib.dump(self.encoders, 'encoders.pkl')
        
        # Save feature importance
        if best_model_name in self.feature_importance:
            joblib.dump(self.feature_importance[best_model_name], 'feature_importance.pkl')
        
        print("✅ Model and preprocessing components saved successfully!")
        
        return best_model
    
    def create_visualizations(self, best_model_name):
        """
        Create comprehensive visualizations
        """
        print("Creating visualizations...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison
        plt.subplot(2, 3, 1)
        comparison_sorted = self.results.sort_values('R2_Score', ascending=True)
        bars = plt.barh(comparison_sorted['Model'], comparison_sorted['R2_Score'], 
                       color='skyblue', alpha=0.7)
        
        # Highlight best model
        best_idx = comparison_sorted[comparison_sorted['Model'] == best_model_name].index[0]
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
        
        plt.xlabel('R² Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 2. RMSE vs MAE
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(self.results['RMSE'], self.results['MAE'], 
                            c=self.results['R2_Score'], cmap='viridis', s=100, alpha=0.7)
        
        # Highlight best model
        best_row = self.results[self.results['Model'] == best_model_name].iloc[0]
        plt.scatter(best_row['RMSE'], best_row['MAE'], c='red', s=200, 
                   marker='*', label='Best Model', edgecolor='black', linewidth=2)
        
        plt.xlabel('RMSE (days)')
        plt.ylabel('MAE (days)')
        plt.title('RMSE vs MAE (colored by R² Score)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='R² Score')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 3. Feature Importance (if available)
        if best_model_name in self.feature_importance:
            plt.subplot(2, 3, 3)
            feature_names = self.X_train.columns
            importance = self.feature_importance[best_model_name]
            
            # Get top 15 features
            top_indices = np.argsort(importance)[-15:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            bars = plt.barh(top_features, top_importance, color='lightcoral', alpha=0.7)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Features - {best_model_name}', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
        
        # 4. Actual vs Predicted
        plt.subplot(2, 3, 4)
        best_model = self.models[best_model_name]
        
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'SVR']:
            X_test_use = self.scalers['standard'].transform(self.X_test)
        else:
            X_test_use = self.X_test
            
        y_pred = best_model.predict(X_test_use)
        
        plt.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Days to Ship')
        plt.ylabel('Predicted Days to Ship')
        plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(self.y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Residuals Plot
        plt.subplot(2, 3, 5)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Days to Ship')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot - {best_model_name}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 6. Distribution of Days to Ship
        plt.subplot(2, 3, 6)
        plt.hist(self.y_test, bins=30, alpha=0.7, color='purple', label='Actual')
        plt.hist(y_pred, bins=30, alpha=0.7, color='orange', label='Predicted')
        plt.xlabel('Days to Ship')
        plt.ylabel('Frequency')
        plt.title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('days_to_ship_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional plot: Cross-validation scores
        plt.figure(figsize=(12, 6))
        cv_means = self.results['CV_R2_Mean']
        cv_stds = self.results['CV_R2_Std']
        
        plt.errorbar(range(len(self.results)), cv_means, yerr=cv_stds, 
                    fmt='o', capsize=5, capthick=2, alpha=0.7)
        plt.xticks(range(len(self.results)), self.results['Model'], rotation=45)
        plt.ylabel('Cross-Validation R² Score')
        plt.title('Cross-Validation Performance with Error Bars', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('cv_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved!")

# Example usage:
def main():
    """
    Main function to run the complete pipeline
    """
    print("🚢 DAYS TO SHIP PREDICTION PIPELINE")
    print("="*50)
    
    # Initialize predictor
    predictor = DaysToShipPredictor()
    
    # Note: Replace this with your actual data loading
    print("📁 Load your data using:")
    print("df = pd.read_csv('your_data.csv')")
    print("\n🔄 Then run the pipeline:")
    print("df_processed = predictor.load_and_preprocess_data(df)")
    print("df_featured = predictor.feature_engineering(df_processed)")
    print("X, y = predictor.prepare_features(df_featured)")
    print("results = predictor.train_models(X, y)")
    print("best_model_name, comparison = predictor.get_model_comparison()")
    print("best_model = predictor.save_best_model(best_model_name)")
    print("predictor.create_visualizations(best_model_name)")
    
    return predictor

# Create example with synthetic data for demonstration
def create_demo_data():
    """
    Create synthetic data for demonstration
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data matching your exact column names
    data = {
        'c0': range(n_samples),  # Index column
        'CompanyCode': np.random.choice(['A', 'B', 'C'], n_samples),
        'BranchCustomerNumber': np.random.randint(1000, 9999, n_samples),
        'CustomerType': np.random.choice(['Retail', 'Wholesale', 'Online'], n_samples),
        'AutoPrintFlag': np.random.choice([0, 1], n_samples),
        'salesOrderWithoutDistribution': np.random.choice([0, 1], n_samples),
        'SalesOrder': [f'SO{i:06d}' for i in range(n_samples)],
        'SalesOrderDate': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples],
        'GovtEndUserType': np.random.choice(['Gov', 'Commercial', 'Educational'], n_samples),
        'PriorityCode': np.random.choice(['High', 'Medium', 'Low'], n_samples),
        'WasTheOrderEverBackordered': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'DateBackordered': [None] * n_samples,  # Initially None, will fill for backordered items
        'LastPrintedDate': [None] * n_samples,
        'PartNumber': [f'PN{i:06d}' for i in range(n_samples)],
        'VendorPartNumber': [f'VPN{i:06d}' for i in range(n_samples)],
        'SkuLength': np.random.uniform(1, 50, n_samples),
        'SkuWidth': np.random.uniform(1, 30, n_samples),
        'SkuHeight': np.random.uniform(1, 20, n_samples),
        'SkuWeight': np.random.uniform(0.1, 10, n_samples),
        'OriginalQuantityOrdered': np.random.randint(1, 100, n_samples),
        'QuantityOpen': np.random.randint(0, 50, n_samples),
        'ShipmentQuantity': np.random.randint(1, 100, n_samples),
        'VendorClass': np.random.choice(['A', 'B', 'C'], n_samples),
        'VendorCode': np.random.choice(['V1', 'V2', 'V3', 'V4'], n_samples),
        'VendorName': np.random.choice(['Vendor1', 'Vendor2', 'Vendor3'], n_samples),
        'MasterVendorCode': np.random.choice(['MV1', 'MV2', 'MV3'], n_samples),
        'MasterVendorName': np.random.choice(['MasterVendor1', 'MasterVendor2'], n_samples),
        'MediaCode': np.random.choice(['CD', 'DVD', 'USB'], n_samples),
        'CategorySubcategory': np.random.choice(['Cat1-Sub1', 'Cat2-Sub1', 'Cat3-Sub2'], n_samples),
        'ProductClass': np.random.choice(['Electronics', 'Hardware', 'Software'], n_samples),
        'NewCpuCode': np.random.choice(['CPU1', 'CPU2', 'CPU3'], n_samples),
        'ProductGroup': np.random.choice(['Group1', 'Group2', 'Group3'], n_samples),
        'UnitCost': np.random.uniform(10, 1000, n_samples),
        'ShipFromBranch': np.random.choice(['Branch1', 'Branch2', 'Branch3'], n_samples),
        'ShipToCity': np.random.choice(['New York', 'Los Angeles', 'Chicago'], n_samples),
        'ShipToState': np.random.choice(['CA', 'NY', 'TX', 'FL'], n_samples),
        'ShipToCountry': np.random.choice(['USA', 'Canada'], n_samples),
        'CarrierCode': np.random.choice(['UPS', 'FedEx', 'DHL'], n_samples),
        'InventoryDate': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples],
        'BalanceOnHand': np.random.randint(0, 500, n_samples),
        'QuantityOnOrder': np.random.randint(0, 200, n_samples),
        'QuantityCommitted': np.random.randint(0, 100, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add DateBackordered for backordered items
    backorder_mask = df['WasTheOrderEverBackordered'] == 1
    backorder_dates = df.loc[backorder_mask, 'SalesOrderDate'] + pd.to_timedelta(
        np.random.randint(1, 10, backorder_mask.sum()), unit='D'
    )
    df.loc[backorder_mask, 'DateBackordered'] = backorder_dates
    
    # Create ActualShipDate based on some logic
    days_to_add = np.random.randint(1, 30, n_samples)  # Random days between 1-30
    
    # Add some logic: backorders take longer, high priority is faster
    for i in range(n_samples):
        base_days = days_to_add[i]
        if df.loc[i, 'WasTheOrderEverBackordered'] == 1:
            base_days += np.random.randint(5, 15)  # Add 5-15 days for backorders
        if df.loc[i, 'PriorityCode'] == 'High':
            base_days = max(1, base_days - 3)  # Subtract 3 days for high priority
        elif df.loc[i, 'PriorityCode'] == 'Low':
            base_days += 2  # Add 2 days for low priority
            
        days_to_add[i] = base_days
    
    df['ActualShipDate'] = df['SalesOrderDate'] + pd.to_timedelta(days_to_add, unit='D')
    
    return df

# Run demo
if __name__ == "__main__":
    # Create demo data
    print("Creating synthetic demo data...")
    demo_df = create_demo_data()
    
    # Run pipeline
    predictor = DaysToShipPredictor()
    df_processed = predictor.load_and_preprocess_data(demo_df)
    df_featured = predictor.feature_engineering(df_processed)
    X, y = predictor.prepare_features(df_featured)
    results = predictor.train_models(X, y)
    best_model_name, comparison = predictor.get_model_comparison()
    best_model = predictor.save_best_model(best_model_name)
    predictor.create_visualizations(best_model_name)