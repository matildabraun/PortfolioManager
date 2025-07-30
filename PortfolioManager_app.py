import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import scipy.stats as stats


#App Title
st.title("Portfolio Optimizer")

# Sidebar Inputs
st.sidebar.header("User Input")
tickers = st.sidebar.text_input("Enter tickers (comma-separated):", "SPY, BND, GLD, QQQ, VTI").upper().split(",")
years = st.sidebar.slider("Select years of historical data:", 1, 10, 5)
# Risk-Free Rate
risk_free_input = st.sidebar.text_input("Risk-Free Rate (e.g., 0.042 for 4.2%)", value="0.042")
try:
    risk_free_input = st.sidebar.text_input("Risk-Free Rate (%)", value="4.2")
try:
    risk_free_rate = float(risk_free_input) / 100  # Convert to decimal
except ValueError:
    st.sidebar.error("Please enter a valid number (e.g., 4.2 for 4.2%)")
    risk_free_rate = 0.042  # Default in case of error

# Download Data
end_date = datetime.today()
start_date = end_date - timedelta(days=years * 365)
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Close']

# Log Returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252


# Optimization Functions
def standard_deviation(weight, cov_matrix):
    return np.sqrt(weight.T @ cov_matrix @ weight)

def expected_returns(weight, log_returns):
    return np.sum(log_returns.mean() * weight) * 252

def sharpe_ratio(weight, log_returns, cov_matrix, risk_free_rate):
    return (expected_returns(weight, log_returns) - risk_free_rate) / standard_deviation(weight, cov_matrix)

def neg_sharpe(weight, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weight, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}
bounds = [(0, 0.5) for _ in range(len(tickers))]
initial_weights = np.array([1 / len(tickers)] * len(tickers))

# Optimization
optimized_results = minimize(neg_sharpe, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                             method='SLSQP', constraints=constraints, bounds=bounds)
optimal_weights = optimized_results.x

# Results
expected_return = expected_returns(optimal_weights, log_returns)
expected_volatility = standard_deviation(optimal_weights, cov_matrix)
sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

# Heatmap of Covariance Matrix (Annualized)
st.subheader("Covariance Matrix")

# Obtener valores mínimos y máximos para normalizar colores
min_val = np.min(cov_matrix.values)
max_val = np.max(cov_matrix.values)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    cov_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=min_val,
    vmax=max_val,
    fmt=".6f",
    linewidths=0.5,
    linecolor="white"
)
st.pyplot(fig)

# Nivel de confianza para VaR
confidence_level = 0.95

# Retorno diario del portafolio (media ponderada)
port_return = np.dot(optimal_weights, log_returns.mean())

# Volatilidad diaria del portafolio (desviación estándar ponderada)
port_volatility = np.sqrt(np.dot(optimal_weights, np.dot(log_returns.cov(), optimal_weights)))

# VaR diario (suponiendo distribución normal)
z_score = stats.norm.ppf(1 - confidence_level)
VaR_daily = -(port_return + z_score * port_volatility)

# Escalar a VaR anual si quieres
VaR_annual = VaR_daily * np.sqrt(252)

# Display Metrics
st.subheader("Portfolio Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Expected Return (Annual)", f"{expected_return:.2%}")
col2.metric("Volatility (Annual)", f"{expected_volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
col4.metric("Daily VaR (95%)", f"{VaR_daily:.4%}")
col5.metric("Anual VaR (95%)", f"{VaR_annual:.2%}")


#Note for interpretation
st.markdown("<small><i>Note: The Sharpe Ratio measures risk-adjusted return — a value above 1 is generally considered good. "
    "Volatility indicates risk; higher volatility implies greater uncertainty. Optimal portfolios aim for high returns with acceptable volatility levels. "
    "Value at Risk (VaR) estimates the maximum expected loss over a given period at a certain confidence level — lower VaR implies less downside risk.</i></small>",
    unsafe_allow_html=True)

# Display Optimal Weights as a Table
st.subheader("Portfolio Allocation")

# Crear DataFrame con todos los tickers y pesos
weights_df = pd.DataFrame({
    'Ticker': tickers,
    'Weight': optimal_weights})

# Convertir pesos a porcentaje, redondear y formatear como string con %
weights_df["Weight"] = (weights_df["Weight"] * 100).round(2).astype(str) + "%"

# Mostrar tabla interactiva sin índice, con ancho completo
st.dataframe(weights_df.reset_index(drop=True), use_container_width=True)

# Pie Chart of Allocation
st.subheader("Allocation Pie Chart")

# Filtrar solo los pesos > 0 para la gráfica
filtered_weights = [(w, t) for w, t in zip(optimal_weights, tickers) if w > 0]

if filtered_weights:
    weights, labels = zip(*filtered_weights)
    fig = px.pie(
    names=labels,
    values=weights,
    hole=0.3,  # estilo donut
    width=700,
    height=700
)

fig.update_traces(textinfo='percent+label')
fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))

st.plotly_chart(fig)


# In[ ]:




