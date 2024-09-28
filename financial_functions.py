import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
from scipy.stats import norm

def existe_ticker(ticker):
    #datosComprobar = yf.Ticker(ticker).history(period="1d")
    return True

def descargar_datos(tickers, fecha_inicio, fecha_fin):
    data = yf.download(tickers, start=fecha_inicio, end=fecha_fin)['Adj Close']
    rendimientos = data.pct_change().dropna()
    return rendimientos, data

def rendimiento_cartera(pesos, rendimientos_medios, matriz_cov, periodo_tiempo=252):
    rendimientos = np.sum(rendimientos_medios * pesos) * periodo_tiempo
    std = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(periodo_tiempo)
    return std, rendimientos

def ratio_sharpe_negativo(pesos, rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    p_var, p_ret = rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)
    return -(p_ret - tasa_libre_riesgo) / p_var

def maximizar_ratio_sharpe(rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    num_activos = len(rendimientos_medios)
    args = (rendimientos_medios, matriz_cov, tasa_libre_riesgo)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = (0.0, 1.0)
    limites = tuple(limite for activo in range(num_activos))

    resultado = minimize(ratio_sharpe_negativo, num_activos*[1./num_activos], args=args,
                         method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado

def volatilidad_cartera(pesos, rendimientos_medios, matriz_cov):
    return rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)[0]

def minima_varianza(rendimientos_medios, matriz_cov):
    num_activos = len(rendimientos_medios)
    args = (rendimientos_medios, matriz_cov)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = (0.0, 1.0)
    limites = tuple(limite for activo in range(num_activos))

    resultado = minimize(volatilidad_cartera, num_activos*[1./num_activos], args=args,
                         method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado

def carteras_aleatorias(num_carteras, rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    resultados = np.zeros((3,num_carteras))
    registro_pesos = []
    for i in range(num_carteras):
        pesos = np.random.random(len(rendimientos_medios))
        pesos /= np.sum(pesos)
        registro_pesos.append(pesos)
        portfolio_std, portfolio_return = rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)
        resultados[0,i] = portfolio_std
        resultados[1,i] = portfolio_return
        resultados[2,i] = (portfolio_return - tasa_libre_riesgo) / portfolio_std
    return resultados, registro_pesos

def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    precio_call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    precio_put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return (precio_call, precio_put)