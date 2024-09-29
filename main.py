import streamlit as st
from ui_components import configure_page, apply_custom_css
from tabs import optimizador_cartera_tab, valoracion_opciones_tab, mapa_calor_opciones_tab, analisis_estadistico_tab,montecarlo_opciones

def main():
    configure_page()
    apply_custom_css()

    st.title("📊 Dashboard Financiero 📊")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Optimizador de Cartera", "Valoración de Opciones","Montecarlo Opciones", "Mapa de Calor Opciones", "Análisis Estadístico" ])

    with tab1:
        optimizador_cartera_tab()

    with tab2:
        valoracion_opciones_tab()

    with tab3:
        montecarlo_opciones()
        

    with tab4:
        mapa_calor_opciones_tab()
        
    with tab5:
        analisis_estadistico_tab()
        


if __name__ == "__main__":
    main()