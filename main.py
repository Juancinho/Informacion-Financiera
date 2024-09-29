import streamlit as st
from ui_components import configure_page, apply_custom_css
from tabs import optimizador_cartera_tab, valoracion_opciones_tab, mapa_calor_opciones_tab, analisis_estadistico_tab

def main():
    configure_page()
    apply_custom_css()

    st.title("📊 Información Financiera 📊")

    tab1, tab2, tab3, tab4 = st.tabs(["Optimizador de Cartera", "Valoración de Opciones", "Mapa de Calor Opciones", "Análisis Estadístico"])

    with tab1:
        optimizador_cartera_tab()

    with tab2:
        valoracion_opciones_tab()

    with tab3:
        mapa_calor_opciones_tab()

    with tab4:
        analisis_estadistico_tab()

if __name__ == "__main__":
    main()