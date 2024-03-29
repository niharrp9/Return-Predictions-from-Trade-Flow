# Return Predictions from Trade Flow

This project explores the potential of trade flow data to forecast returns in the cryptotoken market. It delves into the complexities of transaction-level data, aiming to capture the nuances of trade dynamics and their predictive power over short-term price movements.

## Introduction

The core of this research lies in the analysis of high-frequency trading data from the Coinbase WebSocket API. By dissecting level 3 exchange messages and truncated level 2 data, the project assesses the feasibility of generating actionable insights within three distinct cryptotoken markets.

## Data

The dataset comprises detailed trade and order book information for 2023, preprocessed to facilitate analysis. It's split into training and testing sets, with the first 40% designated for model training and the remainder for testing.

### Data Treatment

Trade flow (Fi(τ)) is calculated for the τ-interval preceding each trade. Subsequently, T-second forward returns (ri(T)) are computed and regressed against the trade flows to determine a coefficient of regression (β), which serves as the foundation for return predictions.

## Exercise

The notebook develops a methodology to estimate trade-induced price changes. It predicts returns as a function of the preceding trade flow, applying thresholds (j) to filter significant predictive opportunities, thereby avoiding overtrading and enhancing strategy efficiency.

## Analysis

In-depth analysis assesses the viability of these predictions. The study investigates key performance metrics such as Sharpe ratios, drawdowns, and tail risks, both with and without trading cost assumptions. The reliability of β is scrutinized, and the impact of parameter choices on the model's predictive capability is explored.

## Getting Started

To interact with the project, clone this repository and launch the Jupyter Notebook:

```sh
git clone https://github.com/your-username/Return-Predictions-Trade-Flow.git
cd Return-Predictions-Trade-Flow
jupyter notebook
