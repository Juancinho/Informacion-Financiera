import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from financial_functions import existe_ticker, descargar_datos, black_scholes
from visualization import mostrar_ef_simulada_con_aleatorias, crear_mapa_calor
from ui_components import plotly_config

def optimizador_cartera_tab():
    st.header("Optimizador de Cartera")
    
    # Ticker input section
    st.subheader("Selección de Tickers")
    
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
    


    # Function to add a ticker
    def add_ticker_func(ticker):
        ticker = ticker.strip().upper()
        if existe_ticker(ticker):
            if ticker not in st.session_state.tickers:
                st.session_state.tickers.append(ticker)
                st.session_state.last_added_ticker = ticker
                st.rerun()
        else:
            st.error(f"No existe el ticker {ticker}")



    # Add new ticker when button is clicked
    if add_ticker and new_ticker:
        add_ticker_func(new_ticker)
    
    # Add new ticker when Enter is pressed
    if new_ticker and new_ticker != st.session_state.last_added_ticker:
        add_ticker_func(new_ticker)
    
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

                ef_fig, max_sharpe_fig, min_vol_fig, max_sharpe_asignacion, min_vol_asignacion,retorno_max_sharpe, retorno_min_vol = mostrar_ef_simulada_con_aleatorias(
                    rendimientos_medios, matriz_cov, num_carteras, tasa_libre_riesgo, tickers
                )

                estadisticas_rendimientos = pd.DataFrame({
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
                    container2 = st.container(border=True)
                    container2.subheader("Cartera con Máximo Ratio de Sharpe")
                    container2.plotly_chart(max_sharpe_fig, use_container_width=True)
                    container2.metric("Retorno Anualizado", f"{retorno_max_sharpe*100:.2f}%")
                    container2.info("Este es el retorno anual esperado basado en datos históricos. No garantiza resultados futuros.")


                with col2:
                    container3 = st.container(border=True)
                    container3.subheader("Cartera de Mínima Volatilidad")
                    container3.plotly_chart(min_vol_fig, use_container_width=True)
                    container3.metric("Retorno Anualizado", f"{retorno_min_vol*100:.2f}%")
                    container3.info("Este es el retorno anual esperado basado en datos históricos. No garantiza resultados futuros.")

                col1, col2 = st.columns(2)
                with col1:
                    container4 = st.container(border=True)
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
                    container5 = st.container(border=True)
                    container5.subheader("Comparación de Rendimiento, Volatilidad y Sharpe")
                    
                    # Crear el gráfico de barras agrupadas
                    rendimientos_fig = make_subplots(rows=1, cols=3, subplot_titles=("Rendimiento Anual", "Volatilidad Anual", "Ratio de Sharpe"))
                    
                    rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Rendimiento Anual'], showlegend=False),row=1, col=1)
                    rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Volatilidad Anual'], showlegend=False),row=1, col=2)
                    rendimientos_fig.add_trace(go.Bar(name='Rendimiento Anual', x=estadisticas_rendimientos.index, y=estadisticas_rendimientos['Ratio de Sharpe'], showlegend=False),row=1, col=3)
                    
                    container5.plotly_chart(rendimientos_fig, use_container_width=True)
                
                container6 = st.container(border=True)
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

def valoracion_opciones_tab():
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

def mapa_calor_opciones_tab():
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

def analisis_estadistico_tab():
    st.header("Análisis Estadístico")
    ticker = st.text_input("Introduce un ticker")

    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio_estadistica = st.date_input("Fecha de Inicio de Datos Históricos", key="fecha_inicio_estadistica")
        
    with col2:
        fecha_fin_estadistica = st.date_input("Fecha de Fin de Datos Históricos", key="fecha_fin_estadistica")

    if st.button("Analizar Distribución de Retornos"):
        if ticker:
            if existe_ticker(ticker):
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
                st.error(f"No existe el ticker {ticker}")
        else:
            st.warning("Por favor, introduce un ticker antes de analizar.")