import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLTradingModel:
    def __init__(self, window_size=5, threshold=0.005):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.window_size = window_size
        self.threshold = threshold

    def _generate_features(self, df):
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_ratio'] = df['sma_10'] / df['sma_50']
        df['volatility'] = df['return'].rolling(10).std()
        df['momentum'] = df['close'] - df['close'].shift(self.window_size)

        df['future_return'] = df['close'].shift(-self.window_size) / df['close'] - 1
        df['label'] = (df['future_return'] > self.threshold).astype(int)

        df.dropna(inplace=True)

        features = df[['return', 'sma_ratio', 'volatility', 'momentum']]
        labels = df['label']
        return features, labels

    def train(self, df):
        X, y = self._generate_features(df)

        # Clean invalid data
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[ML Model] Accuracy: {acc:.4f}")

    def predict(self, df_window):
        X, _ = self._generate_features(df_window)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) == 0:
            return 0.0
        prob = self.model.predict_proba(X.iloc[[-1]])[0][1]
        return prob
