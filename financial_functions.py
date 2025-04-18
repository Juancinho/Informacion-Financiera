import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
from scipy.stats import norm

def existe_ticker(ticker):
    datosComprobar = yf.Ticker(ticker)
    if len(datosComprobar.info)==1:
        return False
    else:
        return True

def descargar_datos(tickers, fecha_inicio, fecha_fin):
    data = yf.download(tickers, start=fecha_inicio, end=fecha_fin)['Close']
    rendimientos = data.pct_change().dropna()
    return rendimientos, data

def rendimiento_cartera(pesos, rendimientos_medios, matriz_cov, periodo_tiempo=252):
    rendimientos = np.sum(rendimientos_medios * pesos) * periodo_tiempo
    std = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(periodo_tiempo)
    return std, rendimientos

def calcular_correlacion_movil(rendimientos, ventana=30):
    correlaciones = rendimientos.rolling(window=ventana).corr()
    return correlaciones



def calcular_retorno_real(pesos, precios):
    """
    Calcula el retorno real de una cartera para un período dado.
    
    :param pesos: Array de pesos de los activos en la cartera
    :param precios: DataFrame de precios históricos
    :return: Retorno total del período
    """
    retornos_diarios = precios.pct_change().dropna()
    retornos_cartera = (retornos_diarios * pesos).sum(axis=1)
    retorno_total = (1 + retornos_cartera).prod() - 1
    return retorno_total


def ratio_sharpe_negativo(pesos, rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    """
    Calcula el ratio de Sharpe negativo para usar en la optimización.
    
    Args:
        pesos: array de pesos de la cartera
        rendimientos_medios: array de rendimientos medios diarios
        matriz_cov: matriz de covarianzas diaria
        tasa_libre_riesgo: tasa libre de riesgo anual
    
    Returns:
        ratio de Sharpe negativo (para minimización)
    """
    # Convertir tasa libre de riesgo anual a diaria
    tasa_libre_riesgo_diaria = (1 + tasa_libre_riesgo)**(1/252) - 1
    
    # Calcular rendimiento y volatilidad anualizados
    p_std, p_ret = rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)
    
    # Calcular ratio de Sharpe
    ratio = (p_ret - tasa_libre_riesgo) / p_std if p_std > 0 else -np.inf
    
    return -ratio  # Negativo porque queremos maximizar

def maximizar_ratio_sharpe(rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    """
    Encuentra la cartera con el máximo ratio de Sharpe.
    
    Args:
        rendimientos_medios: array de rendimientos medios diarios
        matriz_cov: matriz de covarianzas diaria
        tasa_libre_riesgo: tasa libre de riesgo anual
    
    Returns:
        resultado de la optimización
    """
    num_activos = len(rendimientos_medios)
    args = (rendimientos_medios, matriz_cov, tasa_libre_riesgo)
    
    # Restricción: la suma de los pesos debe ser 1
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Límites: los pesos deben estar entre 0 y 1
    limite = (0.0, 1.0)
    limites = tuple(limite for _ in range(num_activos))
    
    # Pesos iniciales equitativos
    pesos_iniciales = np.array([1./num_activos] * num_activos)
    
    # Optimización
    resultado = minimize(
        ratio_sharpe_negativo, 
        pesos_iniciales,
        args=args,
        method='SLSQP',
        bounds=limites,
        constraints=restricciones,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if not resultado.success:
        print("La optimización no convergió:", resultado.message)
    
    return resultado

def rendimiento_cartera(pesos, rendimientos_medios, matriz_cov, periodo_tiempo=252):
    """
    Calcula el rendimiento y la volatilidad anualizada de una cartera.
    
    Args:
        pesos: array de pesos de la cartera
        rendimientos_medios: array de rendimientos medios diarios
        matriz_cov: matriz de covarianzas diaria
        periodo_tiempo: número de períodos para anualizar (252 días por defecto)
    
    Returns:
        tuple de (volatilidad anualizada, rendimiento anualizado)
    """
    # Calcular rendimiento anualizado
    rendimiento = np.sum(rendimientos_medios * pesos) * periodo_tiempo
    
    # Calcular volatilidad anualizada
    volatilidad = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(periodo_tiempo)
    
    return volatilidad, rendimiento

def carteras_aleatorias(num_carteras, rendimientos_medios, matriz_cov, tasa_libre_riesgo):
    """
    Genera carteras aleatorias y calcula sus métricas.
    
    Args:
        num_carteras: número de carteras a generar
        rendimientos_medios: array de rendimientos medios diarios
        matriz_cov: matriz de covarianzas diaria
        tasa_libre_riesgo: tasa libre de riesgo anual
    
    Returns:
        tuple de (resultados, pesos)
        resultados: matriz de (3, num_carteras) con volatilidad, rendimiento y ratio de Sharpe
        pesos: lista de arrays con los pesos de cada cartera
    """
    num_activos = len(rendimientos_medios)
    resultados = np.zeros((3, num_carteras))
    pesos_carteras = []
    
    for i in range(num_carteras):
        # Generar pesos aleatorios que sumen 1
        pesos = np.random.random(num_activos)
        pesos /= np.sum(pesos)
        pesos_carteras.append(pesos)
        
        # Calcular métricas de la cartera
        portfolio_std, portfolio_ret = rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)
        resultados[0,i] = portfolio_std
        resultados[1,i] = portfolio_ret
        resultados[2,i] = (portfolio_ret - tasa_libre_riesgo) / portfolio_std
    
    return resultados, pesos_carteras

def volatilidad_cartera(pesos, rendimientos_medios, matriz_cov):
    return rendimiento_cartera(pesos, rendimientos_medios, matriz_cov)[0]

def minima_varianza(rendimientos_medios, matriz_cov):
    num_activos = len(rendimientos_medios)
    args = (rendimientos_medios, matriz_cov)
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limite = (0.0, 1.0)
    limites = tuple(limite for activo in range(num_activos))

    resultado = minimize(volatilidad_cartera, num_activos*[1./num_activos], args=args,method='SLSQP', bounds=limites, constraints=restricciones)
    return resultado


def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    precio_call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    precio_put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return (precio_call, precio_put)


def montecarlo_black_scholes(S, K, T, r, sigma, num_simulaciones):
    # Generar caminos aleatorios para el precio del activo
    z = np.random.standard_normal(num_simulaciones)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    # Calcular el pago de la opción al vencimiento
    payoffCall = np.maximum(ST - K, 0)
    payoffPut = np.maximum(K - ST, 0)


    # Calcular el precio de la opción (valor presente del pago esperado)
    precioCall = np.exp(-r * T) * np.mean(payoffCall)
    precioPut = np.exp(-r * T) * np.mean(payoffPut)

    return (precioCall,precioPut)



