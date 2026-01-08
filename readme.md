# Intraday Volatility Options Pricing Engine

This project implements a short-term options pricing strategy that leverages Black–Scholes valuation** and VIX–SPY copula dependence modeling to identify volatility dislocations and generate alpha within a single trading day.

This was developed as an introductory quantitative finance project, it integrates concepts from my Financial Engineering coursework in options, derivatives, and computational finance. The goal was to bridge theoretical models with practical data analysis by building a fully functioning options engine capable of testing real-time market dynamics.

## Overview

The model combines financial theory with empirical market data to exploit intraday volatility dynamics:

- **Black–Scholes Valuation** computes theoretical European option prices for SPY calls.
- **VIX Copula Modeling** captures nonlinear dependence between VIX and SPY to detect volatility shifts and identify stronger correlations between SPY and VIS during periods of elevated market volatility.
- **Automated Data Pipeline** fetches and processes market data using `yfinance` and `pandas` and `matplotlib`.
- **Streamlit Dashboard** enables interactive visualization, backtesting, and real-time trade simulation.
