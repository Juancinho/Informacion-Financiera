import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.stats import norm

# Configurar página de Streamlit
st.set_page_config(page_title="Información Financiera", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

# CSS personalizado para mejorar la estética
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #1e2130;
    }
    .Widget>label {
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background: #262730;
        color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background: #262730;
        color: #ffffff;
    }
    .stDateInput>div>div>input {
        background: #262730;
        color: #ffffff;
    }
    .stTab {
        background-color: #1e2130;
        color: #ffffff;
        border-radius: 5px 5px 0 0;
    }
    .stTab[data-baseweb="tab"] {
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTab[aria-selected="true"] {
        background-color: #0e1117;
        border-bottom: 2px solid #FF6F61; /* Salmón */
    }
    .stButton>button {
        background-color: #FF6F61; /* Salmón */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4F3D; /* Salmón oscuro */
    }
    .card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #FF6F61; /* Salmón */
    }
</style>

""", unsafe_allow_html=True)

# Aplicar tema oscuro a todas las figuras de Plotly
plotly_config = {
    'template': 'plotly_dark',

}

# Funciones auxiliares

def existe_ticker(ticker):
    datosComprobar = yf.Ticker(ticker)
    if len(datosComprobar.info)==1:
        return False
    else:
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

# Funciones para Black-Scholes
def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    precio_call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    precio_put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return (precio_call, precio_put)

# Mapa de calor para opciones
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

# Aplicación Streamlit
def main():
    st.title("📊 Información Financiera 📊")

    tab1, tab2, tab3, tab4 = st.tabs(["Optimizador de Cartera", "Valoración de Opciones", "Mapa de Calor Opciones", "Análisis Estadístico"])

    with tab1:
        st.header("Optimizador de Cartera")
        
        # Ticker input section
        st.subheader("Selección de Tickers")
        
        # Custom CSS to adjust input width and center the button
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
           
        }
        .custom-button {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state for tickers and for tracking changes
        if 'tickers' not in st.session_state:
            st.session_state.tickers = []
        if 'last_added_ticker' not in st.session_state:
            st.session_state.last_added_ticker = None
        if 'last_removed_index' not in st.session_state:
            st.session_state.last_removed_index = None
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            new_ticker = st.text_input("Introduce un ticker", key="new_ticker_input")
        with col2:
            st.markdown('<div class="custom-button">', unsafe_allow_html=True)
            add_ticker = st.button("Añadir Ticker")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add new ticker
        if add_ticker and new_ticker:
            new_ticker = new_ticker.strip().upper()
            if not existe_ticker(new_ticker):
                st.error(f"No existe el ticker {new_ticker}")
                pass
                
            else:
                if new_ticker not in st.session_state.tickers:
                    st.session_state.tickers.append(new_ticker)
                    st.session_state.last_added_ticker = new_ticker
                    st.rerun()
                
                
                

        # Display selected tickers
        if st.session_state.tickers:
            st.write("Tickers seleccionados:")
            cols = st.columns(4)  # Create 4 columns for tickers
            for i, ticker in enumerate(st.session_state.tickers):
                with cols[i % 4]:  # Distribute tickers across columns
                    col1, col2 = st.columns([3, 1])
                    col1.write(ticker)
                    if col2.button("❌", key=f"del_{i}", help="Eliminar ticker"):
                        st.session_state.last_removed_index = i
                        st.session_state.tickers.pop(i)
                        st.rerun()
        else:
            st.info("No hay tickers seleccionados. Por favor, añade al menos uno.")

        # Clear input after adding
        if st.session_state.last_added_ticker == new_ticker:
            st.session_state.last_added_ticker = None
            new_ticker = ""

        # Separator
        st.markdown("---")

        # Other parameters
        with st.expander("Parámetros de Optimización", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input("Fecha de Inicio")
                tasa_libre_riesgo = st.number_input("Tasa Libre de Riesgo", value=0.02, step=0.01)
            with col2:
                fecha_fin = st.date_input("Fecha de Fin")
                num_carteras = st.number_input("Número de Carteras a Simular", value=5000, step=1000)

        # Optimization button
        if st.button("Optimizar Cartera", key="boton_optimizar", type="primary"):
            if not st.session_state.tickers:
                st.warning("Por favor, añade al menos un ticker antes de optimizar la cartera.")
            else:
                with st.spinner("Optimizando cartera..."):
                    tickers = st.session_state.tickers
                    rendimientos, precios = descargar_datos(tickers, fecha_inicio, fecha_fin)
                    
                    rendimientos_medios = rendimientos.mean()
                    matriz_cov = rendimientos.cov()

                    ef_fig, max_sharpe_fig, min_vol_fig, max_sharpe_asignacion, min_vol_asignacion = mostrar_ef_simulada_con_aleatorias(
                        rendimientos_medios, matriz_cov, num_carteras, tasa_libre_riesgo, tickers
                    )

                    

                    

                    estadisticas_rendimientos= pd.DataFrame({
                        'Rendimiento Anual': rendimientos_medios * 252,
                        'Volatilidad Anual': rendimientos.std() * np.sqrt(252),
                        'Ratio de Sharpe': (rendimientos_medios * 252 - tasa_libre_riesgo) / (rendimientos.std() * np.sqrt(252))
                    })

                    st.header("Resultados")
                    container = st.container(border=True)

                    container.subheader("Frontera Eficiente")
                    container.plotly_chart(ef_fig, use_container_width=True)

                    st.subheader("Asignaciones de Carteras")
                    col1, col2 = st.columns(2)
                    with col1:
                        container2=st.container(border=True)

                        container2.subheader("Cartera con Máximo Ratio de Sharpe")
                        container2.plotly_chart(max_sharpe_fig, use_container_width=True)

                    with col2:
                        container3=st.container(border=True)
                        container3.subheader("Cartera de Mínima Volatilidad")
                        container3.plotly_chart(min_vol_fig, use_container_width=True)

                    

                    
                    col1, col2 = st.columns(2)
                    with col1:
                        container4=st.container(border=True)

                        container4.subheader("Matriz de Correlación")
                        matriz_correlacion = rendimientos.corr()
                        
                        
                        # Crear mapa de calor para la matriz de correlación
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=matriz_correlacion.values,
                            x=matriz_correlacion.index,
                            y=matriz_correlacion.columns,
                            colorscale='RdBu',
                            zmin=-1,
                            zmax=1,
                            hovertemplate='<b>X</b>: %{x}<br>' +
                                        '<b>Y</b>: %{y}<br>' +
                                        '<b>Correlación</b>: %{z:.2f}<extra></extra>'
                        ))

                        fig_corr.update_layout(
                                                
                            xaxis=dict(constrain='domain'),
                            yaxis=dict(scaleanchor='x', constrain='domain'),
                            dragmode=False,  # Deshabilita el modo de arrastre
                            **plotly_config
                        )
                        

                        container4.plotly_chart(fig_corr, use_container_width=True)
                        
                        

                        
                    with col2:
                        container5=st.container(border=True)
                        container5.subheader("Comparación de Rendimiento, Volatilidad y Sharpe por Ticker")
                        

                        # Crear el gráfico de barras agrupadas
                        rendimientos_fig = make_subplots(rows=1, cols=3, subplot_titles=("Rendimiento Anual", "Volatilidad Anual", "Ratio de Sharpe"))
                        
                        rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Rendimiento Anual']),row=1, col=1)
                        rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Volatilidad Anual']),row=1, col=2)
                        rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Ratio de Sharpe']),row=1, col=3)
                        

                        
                        container5.plotly_chart(rendimientos_fig, use_container_width=True)
                    
                    

                    

                    container6=st.container(border=True)
                    container6.subheader("Histórico de precios")
                    fig_prices = go.Figure()

                    for ticker in tickers:
                        fig_prices.add_trace(go.Scatter(
                            x=precios.index,
                            y=precios[ticker],
                            mode='lines',
                            name=ticker
                        ))

                    fig_prices.update_layout(
                        
                        xaxis_title='Fecha',
                        yaxis_title='Precio',
                        showlegend=True
                    )

                    container6.plotly_chart(fig_prices, use_container_width=True)


    with tab2:
        st.header("Valoración de Opciones")
        st.markdown("Utiliza esta herramienta para valorar opciones financieras usando el modelo de Black-Scholes.")

        with st.expander("Parámetros de la Opción", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                S = st.number_input("Precio Actual del Activo", value=100.0, step=1.0)
                K = st.number_input("Precio de Ejercicio (Strike)", value=100.0, step=1.0)
                T = st.number_input("Tiempo hasta el Vencimiento (años)", value=1.0, step=0.1)
            with col2:
                r = st.number_input("Tasa Libre de Riesgo", value=0.05, step=0.01)
                sigma = st.number_input("Volatilidad (σ)", value=0.2, step=0.01)

        if st.button("Valorar Opción", key="boton_valorar_opcion"):
            precio_call, precio_put = black_scholes(S, K, T, r, sigma)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="card" style="background-color: #4CAF50;">
                        <h3 style="color: white; text-align: center;">Precio de la Opción Call</h3>
                        <h2 style="color: white; text-align: center;">${precio_call:.3f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="card" style="background-color: #f44336;">
                        <h3 style="color: white; text-align: center;">Precio de la Opción Put</h3>
                        <h2 style="color: white; text-align: center;">${precio_put:.3f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with tab3:
        st.header("Mapa de Calor de Precios de Opciones")
        with st.expander("Parámetros del Mapa de Calor", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                S = st.number_input("Precio Actual del Activo", value=100.0, step=1.0, key="mapa_calor_S")
                T = st.number_input("Tiempo hasta el Vencimiento (años)", value=1.0, step=0.1, key="mapa_calor_T")
                r = st.number_input("Tasa Libre de Riesgo", value=0.05, step=0.01, key="mapa_calor_r")
            with col2:
                vol_min, vol_max = st.slider("Rango de Volatilidad", 0.0, 1.0, (0.1, 0.5), step=0.05)
                strike_min, strike_max = st.slider("Rango de Precio de Ejercicio", 50.0, 150.0, (80.0, 120.0), step=5.0)

        if st.button("Generar Mapa de Calor", key="boton_generar_mapa_calor"):
            col1, col2 = st.columns(2)
            with col1:
                fig = crear_mapa_calor(S, T, r, (vol_min, vol_max), (strike_min, strike_max), "call")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = crear_mapa_calor(S, T, r, (vol_min, vol_max), (strike_min, strike_max), "put")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="card">
                <h3>Entendiendo el Mapa de Calor</h3>
                <p>Este mapa de calor muestra cómo cambian los precios de las opciones en función del precio de ejercicio y la volatilidad:</p>
                <ul>
                    <li>Eje X: Precio de Ejercicio</li>
                    <li>Eje Y: Volatilidad</li>
                    <li>Color: Precio de la Opción</li>
                </ul>
                <p>Ajusta los parámetros arriba para ver cómo afectan a los precios de las opciones.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with tab4:
        st.header("Análisis Estadístico")
        ticker = st.text_input("Introduce un ticker")

        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio_estadistica = st.date_input("Fecha de Inicio de Datos Históricos", key="fecha_inicio_estadistica")
            
        with col2:
            fecha_fin_estadistica = st.date_input("Fecha de Fin de Datos Históricos", key="fecha_fin_estadistica")

        if st.button("Analizar Distribución de Retornos"):
            if ticker:
                with st.spinner("Cargando datos y generando análisis..."):
                    rendimientosGraf, _ = descargar_datos(ticker, fecha_inicio_estadistica, fecha_fin_estadistica)
                    
                    # Asegurarse de que rendimientos es una Serie de pandas
                    if isinstance(rendimientosGraf, pd.DataFrame):
                        rendimientosGraf = rendimientosGraf.squeeze()  # Convertir DataFrame a Serie si es necesario
                        print(rendimientosGraf)
                    # Calcular estadísticas descriptivas
                    
                    media = rendimientosGraf.mean()
                    mediana = rendimientosGraf.median()
                    desv_est = rendimientosGraf.std()
                    asimetria = rendimientosGraf.skew()
                    curtosis = rendimientosGraf.kurtosis()
                    
                    # Calcular el número de bins como la raíz cuadrada del número de fechas
                    num_bins = int(np.sqrt(len(rendimientosGraf)))
                    hist, bordes_intervalos = np.histogram(rendimientosGraf, bins=num_bins, density=True)
                    centros_intervalos = (bordes_intervalos[:-1] + bordes_intervalos[1:]) / 2
                    # Crear histograma con distribución normal ajustada
                    mu, std = norm.fit(rendimientosGraf)
                    # Generar puntos para la curva de distribución normal
                    x = np.linspace(rendimientosGraf.min(), rendimientosGraf.max(), 100)
                    p = norm.pdf(x, media, desv_est)

                    histograma = make_subplots(rows=1, cols=1)
                    histograma.add_trace(go.Bar(x=centros_intervalos, y=hist, name='Histograma', marker_color='skyblue', opacity=0.7))
                    histograma.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Ajustada', line=dict(color='red', width=2)))
                  
                    histograma.update_layout(
                        height=800,
                        title=f"Distribución de Retornos Diarios de {ticker}",
                        xaxis_title="Retorno Diario",
                        yaxis_title="Densidad de Probabilidad",
                        showlegend=True,
                        **plotly_config
                    )
                    
                    # Mostrar el gráfico
                    st.plotly_chart(histograma, use_container_width=True)
                    
                    # Mostrar estadísticas descriptivas
                    st.subheader("Estadísticas Descriptivas")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Media", f"{media:.4f}")
                        st.metric("Mediana", f"{mediana:.4f}")
                    with col2:
                        st.metric("Desviación Estándar", f"{desv_est:.4f}")
                        st.metric("Asimetría", f"{asimetria:.4f}")
                    with col3:
                        st.metric("Curtosis", f"{curtosis:.4f}")
                        st.metric("Número de Bins", num_bins)
                    
                    # Mostrar los datos
                    st.subheader("Datos de Retornos")
                    st.dataframe(rendimientosGraf.to_frame(name=ticker))
            else:
                st.warning("Por favor, introduce un ticker antes de analizar.")
          

if __name__ == "__main__":
    main()