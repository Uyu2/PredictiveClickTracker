from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class ClickPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, df):
        X = df.copy()
        
        # Convert timestamp to hour of day and day of week
        X['hour'] = X['timestamp'].dt.hour
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        
        # Encode categorical variables
        categorical_cols = ['search_term', 'device_type', 'browser', 'referrer']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Select features for modeling
        features = ['time_on_screen', 'exited_screen', 'search_count', 'hour', 
                   'day_of_week'] + categorical_cols
        
        return X[features]
    
    def train(self, df):
        X = self.prepare_features(df)
        y = df['clicked']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model.score(X_test, y_test)
    
    def predict(self, df):
        X = self.prepare_features(df)
        return self.model.predict_proba(X)[:, 1]
