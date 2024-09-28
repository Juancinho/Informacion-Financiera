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
    fig2 = go.Figure(go.Pie(labels=max_sharpe_asignacion.columns, values=max_sharpe_asignacion.iloc[0], name='Máximo Ratio de Sharpe'))
    fig2.update_layout(title="Asignación de la Cartera con Máximo Ratio de Sharpe", **plotly_config)

    # Gráfico circular de asignación para la cartera de mínima volatilidad
    fig3 = go.Figure(go.Pie(labels=min_vol_asignacion.columns, values=min_vol_asignacion.iloc[0], name='Mínima Volatilidad'))
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
        colorbar=dict(title='Precio de Opción')
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
    st.title("📊 RAFA ES UN GITANO 📊")

    tab1, tab2, tab3, tab4 = st.tabs(["Optimizador de Cartera", "Valoración de Opciones", "Mapa de Calor Opciones", "Análisis Estadístico"])

    with tab1:
        st.header("Optimizador de Cartera")
        with st.expander("Parámetros de Entrada", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                tickers = st.text_input("Introduce tickers (separados por comas)", "AAPL,GOOGL,MSFT").split(',')
                tickers = [ticker.strip() for ticker in tickers]
                fecha_inicio = st.date_input("Fecha de Inicio")
            with col2:
                fecha_fin = st.date_input("Fecha de Fin")
                tasa_libre_riesgo = st.number_input("Tasa Libre de Riesgo", value=0.02, step=0.01)
                num_carteras = st.number_input("Número de Carteras a Simular", value=5000, step=1000)

        if st.button("Optimizar Cartera", key="boton_optimizar"):
            with st.spinner("Optimizando cartera..."):
                rendimientos, precios = descargar_datos(tickers, fecha_inicio, fecha_fin)
                rendimientos_medios = rendimientos.mean()
                matriz_cov = rendimientos.cov()

                ef_fig, max_sharpe_fig, min_vol_fig, max_sharpe_asignacion, min_vol_asignacion = mostrar_ef_simulada_con_aleatorias(
                    rendimientos_medios, matriz_cov, num_carteras, tasa_libre_riesgo, tickers
                )

                st.header("Resultados")

                st.subheader("Frontera Eficiente")
                st.plotly_chart(ef_fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:

                    st.subheader("Cartera con Máximo Ratio de Sharpe")
                    st.plotly_chart(max_sharpe_fig, use_container_width=True)

                with col2:
                    st.subheader("Cartera de Mínima Volatilidad")
                    st.plotly_chart(min_vol_fig, use_container_width=True)

                st.subheader("Matriz de Correlación")
                matriz_correlacion = rendimientos.corr()
                
                # Crear mapa de calor para la matriz de correlación
                fig_corr = go.Figure(data=go.Heatmap(
                    z=matriz_correlacion.values,
                    x=matriz_correlacion.index,
                    y=matriz_correlacion.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig_corr.update_layout(title="Matriz de Correlación", **plotly_config)

                st.plotly_chart(fig_corr, use_container_width=True)

                st.subheader("Asignaciones de Carteras")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Cartera con Máximo Ratio de Sharpe")
                    st.dataframe(max_sharpe_asignacion)
                with col2:
                    st.write("Cartera de Mínima Volatilidad")
                    st.dataframe(min_vol_asignacion)

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
            st.image("https://statics.memondo.com/p/s1/crs/2022/10/CR_1256595_8c391ae3f82e4dc8a442677d9ba4f93e_escoge_siempre_a_mierdon_thumb_fb.jpg?cb=3630570", width=1500)

if __name__ == "__main__":
    main()