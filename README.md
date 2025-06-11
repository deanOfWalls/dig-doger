# 📈 Dogecoin Trading Strategy

## 🔗 Overview

A systematic trading platform designed to profit from Dogecoin's long-term behavior using a multi-model architecture. The system:

* Buys during downturns
* Exploits local peaks
* Avoids major drawdowns
* Relies exclusively on quantitative models (no discretionary trading)

---

## 💡 Core Philosophy

> "Dogecoin is an asymptotically appreciating asset with temporary downturns. All price declines are opportunities to accumulate."

This principle drives all strategy components:

* Never panic-sell
* Always buy lower and accumulate
* Avoid losses where possible
* Hold through volatility if fundamentals support it

---

## 🔬 Strategy Models

### 1. 🧱 Rules-Based Model

* Hard-coded logic
* Buy after N% drop
* Sell after M% rise
* Uses moving average and peak detection

### 2. 📈 Classic ML Model

* Model: `XGBoostClassifier`
* Features: Momentum, SMA, volatility, volume, etc.
* Target: Binary prediction of profitable 5-minute price moves

### 3. ⚙️ Hybrid Model

* ML for signal generation
* Rule-based constraints for execution
* Combines model intelligence with strategic discipline

### 4. 🧠 Deep RL Agent

* Algorithm: PPO (Proximal Policy Optimization)
* Learns from minute-resolution DOGE price data
* Optimized for ROI, drawdown minimization, and stable gains

---

## 📊 Dataset

* File: `doge_1m_ohlcv.csv`
* Source: Minute-level OHLCV Dogecoin data (2019–Present)
* Format: `timestamp, open, high, low, close, volume, trade_count`

---

## 💰 Capital Management

Tracks the following per trade:

* Fiat balance
* Total DOGE held
* Average cost basis
* ROI (per trade and cumulative)
* Realized vs unrealized gains
* FIFO ledger for tax tracking
* Estimated taxes (35-40% on short-term gains)
* Exchange fees (configurable)

---

## 📅 Evaluation Metrics

* ROI
* Sharpe Ratio
* Max Drawdown
* Trade frequency
* Capital efficiency
* Equity curve analysis

---

## 🔹 Project Goals

* Fully automated, zero-emotion trading
* Faith in DOGE as a long-term appreciating asset
* All dips are opportunities
* Prioritize avoiding losses
* Extensive model testing with real + synthetic data
* Built for clarity, extensibility, and testability
