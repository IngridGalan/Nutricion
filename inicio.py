import stramlit as st
import stramlit as st
import os

st.set_page_config(layout="wide")

st.title("Proyecto Final")

st.markdown("""
## BIENVENIDO
""")

col1, col2 = st.columns([2,2])

with col1:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("EDA: Análisis exploratorio de datos")
    st.markdown("Examina los datos y descubre patrones interesantes")

col3, col4 = st.columns([2,2])

with col3:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.subheader("HIPOTESIS")
    st.markdown("Evalua diferentes hipotesis mediante gráficos")

col5, col6 = st.columns([2,2])

with col5:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.subheader("MODELO")
    st.markdown("Evalua diferentes hipotesis mediante gráficos")