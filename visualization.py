import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from financial_functions import calcular_retorno_real,rendimiento_cartera, maximizar_ratio_sharpe, minima_varianza, carteras_aleatorias, black_scholes
from ui_components import plotly_config

def mostrar_ef_simulada_con_aleatorias(rendimientos_medios, matriz_cov, num_carteras, tasa_libre_riesgo, tickers,precios):
    resultados, pesos = carteras_aleatorias(num_carteras, rendimientos_medios, matriz_cov, tasa_libre_riesgo)

    max_sharpe = maximizar_ratio_sharpe(rendimientos_medios, matriz_cov, tasa_libre_riesgo)
    sdp, rp = rendimiento_cartera(max_sharpe['x'], rendimientos_medios, matriz_cov)
    retorno_real_max_sharpe = calcular_retorno_real(max_sharpe['x'], precios)
    max_sharpe_asignacion = pd.DataFrame(max_sharpe['x'], index=tickers, columns=['asignación'])
    max_sharpe_asignacion['asignación'] = [round(i*100, 2) for i in max_sharpe_asignacion['asignación']]
    max_sharpe_asignacion = max_sharpe_asignacion.T

    min_vol = minima_varianza(rendimientos_medios, matriz_cov)
    sdp_min, rp_min = rendimiento_cartera(min_vol['x'], rendimientos_medios, matriz_cov)
    retorno_real_min_vol = calcular_retorno_real(min_vol['x'], precios)
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
    fig1.update_layout(xaxis_title="Riesgo Anualizado", yaxis_title="Rendimiento Anualizado", **plotly_config)

    # Gráfico circular de asignación para la cartera con el máximo ratio de Sharpe
    fig2 = go.Figure(go.Pie(labels=max_sharpe_asignacion.columns, values=max_sharpe_asignacion.iloc[0], name='Máximo Ratio de Sharpe', hole=.6))
    fig2.update_traces(textposition='outside',textfont_size=22)
    

    # Gráfico circular de asignación para la cartera de mínima volatilidad
    fig3 = go.Figure(go.Pie(labels=min_vol_asignacion.columns, values=min_vol_asignacion.iloc[0], name='Mínima Volatilidad', hole=.6))
    fig3.update_traces(textposition='outside',textfont_size=22)
    

    return fig1, fig2, fig3, max_sharpe_asignacion, min_vol_asignacion, rp, rp_min, retorno_real_max_sharpe, retorno_real_min_vol



def crear_grafico_correlacion_tiempo(correlaciones, tickers):
    fig = go.Figure()
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            ticker1, ticker2 = tickers[i], tickers[j]
            corr = correlaciones.xs(key=ticker2, level=1)[ticker1]
            fig.add_trace(go.Scatter(
                x=corr.index,
                y=corr.values,
                mode='lines',
                name=f'{ticker1} vs {ticker2}'
            ))
    
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Correlación",
        yaxis=dict(range=[-1, 1]),
        **plotly_config
    )
    
    return fig







def graficar_caminos_montecarlo(S, T, r, sigma, num_simulaciones, num_steps):
    """
    Genera y grafica los primeros 5 caminos simulados para el precio del activo,
    incluyendo la banda de confianza del 95%.
    """
    # Tiempo discreto
    dt = T / num_steps
    t = np.linspace(0, T, num_steps)

    # Simulaciones de Monte Carlo para los caminos
    paths = np.zeros((num_steps, num_simulaciones))
    paths[0] = S

    # Simulación de los caminos
    for i in range(1, num_steps):
        z = np.random.standard_normal(num_simulaciones)
        paths[i] = paths[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Cálculo de la banda de confianza
    mean_ln_st = np.log(S) + (r - 0.5 * sigma**2) * t
    std_ln_st = sigma * np.sqrt(t)

    #Banda de desviación típica

    lower_band = np.exp(mean_ln_st - std_ln_st)
    upper_band = np.exp(mean_ln_st + std_ln_st)

    # Banda de confianza al 95%
    #lower_band = np.exp(mean_ln_st - 1.96 * std_ln_st) #A ojo es 1.96 ma o meno
    #upper_band = np.exp(mean_ln_st + 1.96 * std_ln_st)

    # Crear una figura de Plotly
    fig = go.Figure()


    fig.update_layout(
        autosize=False,
        width=1200,
        height=800,
    )


    # Añadir la banda de confianza
    fig.add_trace(go.Scatter(
        x=t,
        y=lower_band,
        fill=None,
        mode='lines',
        line_color='rgba(128, 128, 128, 0.1)',  # Gris claro con opacidad
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=t,
        y=upper_band,
        fill='tonexty',
        mode='lines',
        line_color='rgba(128, 128, 128, 0.1)',  # Gris claro con opacidad
        fillcolor='rgba(128, 128, 128, 0.08)',  # Gris claro con mayor transparencia
        name='Banda de desviación típica 68,27%'
    ))

    # Añadir las simulaciones
    for i in range(num_simulaciones):
        fig.add_trace(go.Scatter(
            x=t,
            y=paths[:, i],
            mode='lines',
            name=f'Simulación {i+1}'
        ))



    # Añadir el precio inicial
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[S, S],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Precio inicial del activo'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title='Caminos simulados del precio del activo',
        xaxis_title='Tiempo (años)',
        yaxis_title='Precio del activo',
        legend_title='Simulaciones',
        template='plotly_white',
        legend=dict(x=0.76, y=1.15, xanchor='left', yanchor='top')

    )

    # Mostrar la gráfica
    return fig


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