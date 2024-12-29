st.title("Analisis Exploratioro de Datos")

if = load_data()

st.header("Aspectos Básicos del Conjunto de Datos")
with st.container():
    col1, col2, col3 = st.columns (3)
    with col1:
        st.metric(label="Numero de Filas", value=df.shape[0], border=True)
    with col2:
         st.metric(label="Numero de Columnas", value=df.shape[1], border=True)
    with col3:
        missing_values = df.isnull().any().sum()
         st.metric(label="Valores pedidos", value= "si" if missing_values > 0 else "No", border=True)

scatter_fig = plot_scatter(df, x_colum, y_column)
st.ploty_chart(scatter_fig)