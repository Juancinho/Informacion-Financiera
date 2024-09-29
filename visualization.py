import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from financial_functions import rendimiento_cartera, maximizar_ratio_sharpe, minima_varianza, carteras_aleatorias, black_scholes
from ui_components import plotly_config

def mostrar_ef_simulada_con_aleatorias(rendimientos_medios, matriz_cov, num_carteras, tasa_libre_riesgo, tickers):
    resultados, pesos = carteras_aleatorias(num_carteras, rendimientos_medios, matriz_cov, tasa_libre_riesgo)

    max_sharpe = maximizar_ratio_sharpe(rendimientos_medios, matriz_cov, tasa_libre_riesgo)
    sdp, rp = rendimiento_cartera(max_sharpe['x'], rendimientos_medios, matriz_cov)
    max_sharpe_asignacion = pd.DataFrame(max_sharpe['x'], index=tickers, columns=['asignación'])
    max_sharpe_asignacion['asignación'] = [round(i*100, 2) for i in max_sharpe_asignacion['asignación']]
    max_sharpe_asignacion = max_sharpe_asignacion.T

    min_vol = minima_varianza(rendimientos_medios, matriz_cov)
    sdp_min, rp_min = rendimiento_cartera(min_vol['x'], rendimientos_medios, matriz_cov)
    min_vol_asignacion = pd.DataFrame(min_vol['x'], index=tickers, columns=['asignación'])
    min_vol_asignacion['asignación'] = [round(i*100, 2) for i in min_vol_asignacion['asignación']]
    min_vol_asignacion = min_vol_asignacion.T

    # Frontera Eficiente con Carteras Aleatorias
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=resultados[0, :],
        y=resultados[1, :],
        mode='markers',
        marker=dict(size=5, color=resultados[2, :], colorscale='Viridis', showscale=True),
        text=[f'Sharpe: {s:.2f}' for s in resultados[2, :]],
        name='Carteras Simuladas'
    ))
    fig1.add_trace(go.Scatter(x=[sdp], y=[rp], mode='markers', marker=dict(color='red', size=10, symbol='star'), name='Cartera con Máximo Ratio de Sharpe'))
    fig1.add_trace(go.Scatter(x=[sdp_min], y=[rp_min], mode='markers', marker=dict(color='green', size=10, symbol='star'), name='Cartera de Mínima Volatilidad'))
    fig1.update_layout(title="Frontera Eficiente", xaxis_title="Riesgo Anualizado", yaxis_title="Rendimiento Anualizado", **plotly_config)

    # Gráfico circular de asignación para la cartera con el máximo ratio de Sharpe
    fig2 = go.Figure(go.Pie(labels=max_sharpe_asignacion.columns, values=max_sharpe_asignacion.iloc[0], name='Máximo Ratio de Sharpe', hole=.6))
    fig2.update_traces(textposition='outside',textfont_size=22)
    fig2.update_layout(title="Asignación de la Cartera con Máximo Ratio de Sharpe", **plotly_config)

    # Gráfico circular de asignación para la cartera de mínima volatilidad
    fig3 = go.Figure(go.Pie(labels=min_vol_asignacion.columns, values=min_vol_asignacion.iloc[0], name='Mínima Volatilidad', hole=.6))
    fig3.update_traces(textposition='outside',textfont_size=22)
    fig3.update_layout(title="Asignación de la Cartera de Mínima Volatilidad", **plotly_config)

    return fig1, fig2, fig3, max_sharpe_asignacion, min_vol_asignacion

def crear_mapa_calor(S, T, r, rango_volatilidad, rango_strike, tipo_opcion='call'):
    volatilidades = np.linspace(rango_volatilidad[0], rango_volatilidad[1], 20)
    strikes = np.linspace(rango_strike[0], rango_strike[1], 20)
    
    precios_opcion = np.zeros((len(volatilidades), len(strikes)))
    
    for i, sigma in enumerate(volatilidades):
        for j, K in enumerate(strikes):
            precio_call, precio_put = black_scholes(S, K, T, r, sigma)
            precios_opcion[i, j] = precio_call if tipo_opcion == 'call' else precio_put
    
    fig = go.Figure(data=go.Heatmap(
        z=precios_opcion,
        x=strikes,
        y=volatilidades,
        colorscale='rdylgn',
        colorbar=dict(title='Precio de Opción'),
        hovertemplate='<b>Strike</b>: %{x}<br>' +
                                    '<b>Volatilidad</b>: %{y}<br>' +
                                    '<b>Precio</b>: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        width=800,
        height=800,
        title=f'Precio de Opción {tipo_opcion.capitalize()}',
        xaxis_title='Strike',
        yaxis_title='Volatilidad',
        **plotly_config
    )
    
    return fig