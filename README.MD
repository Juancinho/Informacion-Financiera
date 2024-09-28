# 📊 Dashboard de Información Financiera 📈

Bienvenido al proyecto Dashboard de Información Financiera, una aplicación web interactiva que proporciona herramientas potentes para el análisis financiero, la optimización de carteras y la valoración de opciones.

## 🌟 Características

### 1. Optimizador de Cartera

- Optimiza tu cartera de inversión utilizando la teoría moderna de carteras.
- Visualiza la frontera eficiente con carteras simuladas.
- Obtén asignaciones óptimas para la cartera con el máximo ratio de Sharpe y la de mínima volatilidad.
- Analiza la matriz de correlación entre los activos seleccionados.
- Compara rendimientos anuales, volatilidad y ratios de Sharpe entre los activos.
- Visualiza el histórico de precios de los activos seleccionados.

### 2. Valoración de Opciones

- Calcula los precios de opciones utilizando el modelo Black-Scholes.
- Obtén valoraciones para opciones call y put.

### 3. Mapa de Calor de Precios de Opciones

- Visualiza cómo cambian los precios de las opciones en función del precio de ejercicio y la volatilidad.
- Genera mapas de calor para opciones call y put.

### 4. Análisis Estadístico

- Analiza la distribución de retornos de acciones individuales.
- Visualiza histogramas de retornos con distribución normal ajustada.
- Obtén estadísticas descriptivas clave como media, mediana, desviación estándar, asimetría y curtosis.

## 🚀 Comenzando

### Prerrequisitos

- Python 3.7+
- pip

### Instalación

1. Clona el repositorio:
   
   ```
   git clone https://github.com/tuusuario/dashboard-financiero.git
   cd dashboard-financiero
   ```

2. Instala los paquetes requeridos:
   
   ```
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación Streamlit:
   
   ```
   streamlit run app.py
   ```

4. Abre tu navegador web y navega a `http://localhost:8501`.

## 💻 Uso

### Optimizador de Cartera

1. Ingresa los tickers de las acciones en el campo de entrada.
2. Establece el rango de fechas para el análisis histórico.
3. Ajusta la tasa libre de riesgo y el número de carteras a simular.
4. Haz clic en "Optimizar Cartera" para ver:
   - La frontera eficiente con carteras simuladas.
   - Asignaciones óptimas para la cartera con máximo ratio de Sharpe y mínima volatilidad.
   - Matriz de correlación entre los activos.
   - Comparación de rendimientos, volatilidad y ratios de Sharpe por activo.
   - Histórico de precios de los activos seleccionados.

### Valoración de Opciones

1. Ingresa los parámetros de la opción:
   - Precio actual del activo
   - Precio de ejercicio (Strike)
   - Tiempo hasta el vencimiento (en años)
   - Tasa libre de riesgo
   - Volatilidad
2. Haz clic en "Valorar Opción" para ver los precios calculados para las opciones call y put.

### Mapa de Calor de Precios de Opciones

1. Establece los parámetros para el activo y los rangos de volatilidad y precios de ejercicio.
2. Haz clic en "Generar Mapa de Calor" para visualizar cómo cambian los precios de las opciones con diferentes volatilidades y strikes.
3. Interpreta los mapas de calor para opciones call y put.

### Análisis Estadístico

1. Ingresa el ticker de una acción y establece el rango de fechas para el análisis.
2. Haz clic en "Analizar Distribución de Retornos" para ver:
   - Histograma de retornos con distribución normal ajustada.
   - Estadísticas descriptivas clave (media, mediana, desviación estándar, asimetría, curtosis).
   - Tabla de datos de retornos.

## 🛠️ Tecnologías Utilizadas

- [Streamlit](https://streamlit.io/): Para crear la aplicación web interactiva.
- [yfinance](https://pypi.org/project/yfinance/): Para obtener datos financieros históricos.
- [Plotly](https://plotly.com/): Para gráficos y visualizaciones interactivas.
- [NumPy](https://numpy.org/) y [SciPy](https://www.scipy.org/): Para cálculos numéricos y estadísticos.
- [Pandas](https://pandas.pydata.org/): Para manipulación y análisis de datos.

## 📊 Detalles Técnicos

### Optimización de Cartera

- Utiliza el algoritmo de optimización SLSQP (Sequential Least Squares Programming) para maximizar el ratio de Sharpe y minimizar la varianza.
- Simula carteras aleatorias para visualizar la frontera eficiente.
- Calcula y visualiza la matriz de correlación entre los activos seleccionados.

### Modelo Black-Scholes

- Implementa el modelo Black-Scholes para la valoración de opciones europeas.
- Utiliza la función de distribución normal acumulativa para los cálculos.

### Análisis Estadístico

- Calcula estadísticas descriptivas como media, mediana, desviación estándar, asimetría y curtosis.
- Ajusta una distribución normal a los retornos históricos para comparación visual.

## 🎨 Diseño de la Interfaz

- Utiliza un tema oscuro para mejor visibilidad de los gráficos financieros.
- Implementa un diseño responsivo con columnas para una mejor organización de la información.
- Utiliza contenedores con bordes para separar visualmente diferentes secciones de información.

## 📈 Mejoras Futuras

- Agregar más estrategias de optimización de carteras.
- Implementar simulaciones de Monte Carlo para análisis de riesgo.
- Incluir herramientas de análisis fundamental.
- Añadir soporte para datos de criptomonedas.
- Implementar backtesting de estrategias de trading.
- Agregar más modelos de valoración de opciones (ej. Monte Carlo, árboles binomiales).
- Mejorar la velocidad de carga y procesamiento de datos para grandes conjuntos de datos.

## 

## 📝 Licencia

Este proyecto está bajo la licencia [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).

## 👥 Autores

- **Joao** -  [Juancinho](https://github.com/Juancinho)

## 🙏 Agradecimientos

- Agradecimiento a la comunidad de código abierto por las bibliotecas utilizadas.

## 📚 Recursos Adicionales

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Guía de yfinance](https://pypi.org/project/yfinance/)
- [Documentación de Plotly](https://plotly.com/python/)
- [Modelo Black-Scholes](https://www.investopedia.com/terms/b/blackscholes.asp)

---

Hecho con ❤️ y ☕
