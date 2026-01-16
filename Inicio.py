import streamlit as st
from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR, DATA_DIR
import openpyxl 
import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Reporte Capacitaciones Cliente 6382",
    layout="wide",
    page_icon="📊"
)

st.title("📚 Reporte de Capacitaciones - 6382")

#------------------------------------------------------------------------------------------------------------
# Carga de datasets

def cargar_datos(tipo):

    if tipo == "usuarios":
        ruta = DATA_DIR / "usuarios"
    elif tipo == "capacitaciones":
        ruta = DATA_DIR / "capacitaciones"
    else:
        raise ValueError(f"Tipo desconocido: {tipo}")

    archivos = list(Path(ruta).glob("*.xlsx"))
    dfs = [pd.read_excel(archivo, sheet_name="result") for archivo in archivos]
    df = pd.concat(dfs, ignore_index=True)

    # Normalizar columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("___", "_")
        .str.replace("&", "and")
        .map(lambda x: unidecode.unidecode(x))
    )

    # Convertir tipos si existen las columnas
    if 'userscore' in df.columns:
        df['userscore'] = pd.to_numeric(df['userscore'], errors='coerce')
    if 'date' in df.columns:
        df['Fecha'] = pd.to_datetime(df['date'], errors='coerce')

    return df

#------------------------------------------------------------------------------------------------------------
# Procesamiento de tablas

def procesar_datos(df_usuarios, df_capacitaciones):
   
    usuarios = df_usuarios.drop_duplicates()

    training_ids = df_capacitaciones['trainingid'].unique()
    training_ids.sort()

    estados_list = []

    for tid in training_ids:
        df_tid = df_capacitaciones[df_capacitaciones['trainingid'] == tid]

        max_scores = df_tid.groupby('userid')['userscore'].max().reset_index()

        def estado_score(score):
            if pd.isna(score):
                return "Pendiente"
            elif score >= 8:
                return "Aprobado"
            else:
                return "Reprobado"

        max_scores['estado'] = max_scores['userscore'].apply(estado_score)

        df_estado = usuarios.merge(max_scores[['userid', 'estado']], on='userid', how='left')

        df_estado['estado'] = df_estado['estado'].fillna("Pendiente")
        df_estado['Fecha'] = df_tid.groupby('userid')['Fecha'].max().reindex(df_estado['userid']).values

        training_title = df_tid['trainingtitle'].iloc[0] if not df_tid.empty else f"Training {tid}"
        df_estado['training_id'] = tid
        df_estado['training_title'] = training_title

        estados_list.append(df_estado)

    df_final = pd.concat(estados_list, ignore_index=True)

    # Agregar columna 'Capacitado': True si Estado es Aprobado o Reprobado, False si Pendiente
    df_final['Capacitado'] = df_final['estado'].isin(['Aprobado', 'Reprobado'])

    return df_final

#------------------------------------------------------------------------------------------------------------
# Función para cargar y procesar datos (llamada al inicio o al actualizar)
@st.cache_data(show_spinner=False)

def cargar_y_procesar():
    df_usuarios = cargar_datos("usuarios")
    df_capacitaciones = cargar_datos("capacitaciones")
    df_final = procesar_datos(df_usuarios, df_capacitaciones)

   # Fecha actual para historico
    fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nombre del archivo con fecha
    archivo_historico = PROCESSED_DIR / f"capacitaciones_procesadas_{fecha_str}.csv"

    df_final.to_csv(archivo_historico, index=False)

    return df_final

#------------------------------------------------------
# Lógica de persistencia de datos
def obtener_ultimo_procesado():
    archivos = list(PROCESSED_DIR.glob("capacitaciones_procesadas_*.csv"))
    if not archivos:
        return None
    # Retorna el archivo más reciente basado en el nombre (gracias al formato YYYYMMDD_HHMMSS)
    return max(archivos, key=lambda x: x.name)

# Botón para forzar actualización
if st.button("🔄 Actualizar Datos"):
    st.info("Cargando y procesando nuevos archivos...")
    df_final = cargar_y_procesar()
    st.success("✅ Datos actualizados correctamente!")
else:
    ultimo_archivo = obtener_ultimo_procesado()
    if ultimo_archivo:
        # Cargar el último generado
        df_final = pd.read_csv(ultimo_archivo)
        # Asegurar que la fecha vuelva a ser datetime tras leer el CSV
        if 'Fecha' in df_final.columns:
            df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])
    else:
        st.info("Cargando datos por primera vez...")
        df_final = cargar_y_procesar()
        st.success("✅ Datos cargados correctamente!")

#------------------------------------------------------------------------------------------------------------
# Visualización con Streamlit

# Mostrar filtros para segmentación
st.sidebar.header("Filtros")

filtro_puesto = st.sidebar.selectbox("Puesto", options=["Todos"]  + list(df_final['puesto'].dropna().unique()))
filtro_categoria = st.sidebar.selectbox("Categoria", options=["Todos"]  + list(df_final['categoria'].dropna().unique()))
filtro_vicepresidencia = st.sidebar.selectbox("Vicepresidencia", options=["Todos"] + list(df_final['vicepresidencia'].dropna().unique()))
filtro_division = st.sidebar.selectbox("Division", options=["Todos"] + list(df_final['division'].dropna().unique()))
filtro_marca_area = st.sidebar.selectbox("Marca / Area", options=["Todos"] + list(df_final['marca_area'].dropna().unique()))
filtro_centro_trabajo = st.sidebar.selectbox("Centro de Trabajo", options=["Todos"] + list(df_final['centro_de_trabajo'].dropna().unique()))

# Aplicar filtros
def aplicar_filtros(df):
  
    # Filtros por categoría
    if filtro_puesto != "Todos":
        df = df[df['puesto'] == filtro_puesto]
    if filtro_categoria != "Todos":
        df = df[df['categoria'] == filtro_categoria]
    if filtro_vicepresidencia != "Todos":
        df = df[df['vicepresidencia'] == filtro_vicepresidencia]
    if filtro_division != "Todos":
        df = df[df['division'] == filtro_division]
    if filtro_marca_area != "Todos":
        df = df[df['marca_area'] == filtro_marca_area]
    if filtro_centro_trabajo != "Todos":
        df = df[df['centro_de_trabajo'] == filtro_centro_trabajo]
       
    return df

df_filtrado = aplicar_filtros(df_final).rename(columns={'userid': 'ID Usuario','usuario': 'Usuario','puesto': 'Puesto', 'categoria': 'Categoria', 'vicepresidencia': 'Vicepresidencia',
                                                        'division': 'Division', 'marca_area': 'Marca / Area', 'centro_de_trabajo': 'Centro de Trabajo', 
                                                        'training_title': 'Capacitación','estado': 'Estado'})

#------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# Mostrar sección global si no hay filtros aplicados
mostrar_seccion_global = all([
    filtro_puesto == "Todos",
    filtro_categoria == "Todos",
    filtro_vicepresidencia == "Todos",
    filtro_division == "Todos",
    filtro_marca_area == "Todos",
    filtro_centro_trabajo == "Todos",
    ])
if mostrar_seccion_global:

    st.markdown("## 🌎 Visión Global por Centro y Área")
    st.markdown(
        "Esta sección se muestra cuando **no hay filtros aplicados**, "
        "permitiendo comparar centros de trabajo y ver participación por área y centro."
    )


    # -------------------- RESUMEN GENERAL --------------------
    # Total de empleados únicos según filtros
    usuarios_totales = df_filtrado['ID Usuario'].nunique()

    # Usuarios capacitados (según columna Capacitado)
    usuarios_cap_df = df_filtrado[df_filtrado['Capacitado']]
    usuarios_con_cap = usuarios_cap_df['ID Usuario'].nunique()
    usuarios_sin_cap = usuarios_totales - usuarios_con_cap

    st.markdown("### 🥧 Distribución de Empleados")
    st.markdown(f"#### 👤 Total de empleados: {usuarios_totales:,}")

    # -------------------- COLUMNAS --------------------
    col1, col2 = st.columns([1,1])

    # -------------------- COLUMNA 1: Con / Sin Capacitación --------------------
    with col1:
        st.markdown(f"**Empleados con capacitación:** {usuarios_con_cap:,}")
        st.markdown(f"**Empleados sin capacitación:** {usuarios_sin_cap:,}")

        labels = ['Con capacitación', 'Sin capacitación']
        sizes = [usuarios_con_cap, usuarios_sin_cap]
        colors = ['#FF8C00', '#FFCC99']  # naranja fuerte y claro

        plt.figure(figsize=(4,4))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'weight':'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
        plt.axis('equal')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    # -------------------- COLUMNA 2: Aprobados / Reprobados --------------------
    with col2:
        # Contar usuarios únicos aprobados y reprobados
        usuarios_estado = (
            usuarios_cap_df.groupby('ID Usuario')['Estado']
            .max()  # Si un usuario tiene varias filas, toma Aprobado > Reprobado
            .value_counts()
        )

        st.markdown(f"**Empleados aprobados:** {usuarios_estado.get('Aprobado', 0):,}")
        st.markdown(f"**Empleados reprobados:** {usuarios_estado.get('Reprobado', 0):,}")

        labels = usuarios_estado.index.tolist()
        sizes = usuarios_estado.values
        colors = ['#8CCFE0', '#FF6347']  # azul y rojo tomate

        plt.figure(figsize=(4,4))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'weight':'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
        plt.axis('equal')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    # -------------------- CAPACITACIONES POR MES --------------------
    st.markdown("### 📅 Capacitaciones por Año y Mes")
    df_filtrado['mes'] = df_filtrado['Fecha'].dt.to_period('M').dt.to_timestamp()
    mes_df = df_filtrado.groupby('mes').size().reset_index(name='cantidad')

    plt.figure(figsize=(14,6))
    sns.set_style("whitegrid")
    sns.lineplot(data=mes_df, x='mes', y='cantidad', marker='o', color='#FF8C00', linewidth=2.5)
    plt.xticks(rotation=45)
    plt.ylabel("Cantidad de capacitaciones")
    plt.xlabel("Mes")
    plt.title("Capacitaciones realizadas por Mes", fontsize=14, fontweight='bold')

    for x, y in zip(mes_df['mes'], mes_df['cantidad']):
        plt.text(x, y + 2, str(y), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------- CAPACITACIONES POR AÑO Y TRIMESTRE --------------------
    st.markdown("### 📅 Capacitaciones por Año y Trimestre")

    # Aseguramos que los valores sean enteros y manejamos posibles nulos con .fillna(0)
    df_filtrado['Año'] = df_filtrado['Fecha'].dt.year.fillna(0).astype(int)
    df_filtrado['Trimestre'] = df_filtrado['Fecha'].dt.quarter.fillna(0).astype(int)

    # Agrupamos y quitamos los registros donde el año es 0 (fechas inválidas)
    cap_periodo = (
        df_filtrado[df_filtrado['Año'] > 0]
        .groupby(['Año', 'Trimestre'])
        .size()
        .reset_index(name='Cantidad')
    )

    # Formateamos el texto sin puntos decimales
    cap_periodo['Periodo'] = (
        cap_periodo['Año'].astype(str) + 
        " - Q" + 
        cap_periodo['Trimestre'].astype(str)
    )

    # Mostramos la tabla limpia
    st.dataframe(
        cap_periodo[['Periodo', 'Cantidad']].set_index('Periodo'),
        use_container_width=True
    )

# -------------------- CANTIDAD DE EMPLEADOS POR CAPACITACIÓN --------------------
    st.markdown("### 📊 Cantidad de Empleados por Capacitación")

    # Filtrar solo registros de empleados capacitados
    df_cap_ac = df_filtrado[df_filtrado['Capacitado']].copy()

    # Asegurar formato limpio de Año y Trimestre para la tabla
    df_cap_ac['Año'] = df_cap_ac['Fecha'].dt.year.fillna(0).astype(int)
    df_cap_ac['Trimestre'] = df_cap_ac['Fecha'].dt.quarter.fillna(0).astype(int)
    
    # Creamos la etiqueta de periodo para agrupar
    df_cap_ac['Periodo'] = [
        f"{int(a)} - Q{int(t)}" if a > 0 else "Sin Fecha" 
        for a, t in zip(df_cap_ac['Año'], df_cap_ac['Trimestre'])
    ]

    # Agrupar por Capacitación y Periodo
    cap_count = (
        df_cap_ac[df_cap_ac['Año'] > 0]
        .groupby(['Periodo', 'Capacitación'])['ID Usuario']
        .nunique()
        .reset_index(name='Cantidad de Empleados')
        .sort_values(['Periodo', 'Cantidad de Empleados'], ascending=[True, False])
    )

    # Mostrar la tabla con el nuevo desglose
    st.dataframe(
        cap_count.style.set_properties(**{'text-align': 'center'}),
        use_container_width=True
    )

    # -------------------- TOP CENTROS CON MENOR PARTICIPACIÓN --------------------
    st.markdown("## Centros con Menor Participación en Capacitaciones")

    # Base única de empleados por centro
    base_empleados = df_filtrado[['ID Usuario', 'Centro de Trabajo']].drop_duplicates()

    # Empleados capacitados por centro
    empleados_capacitados = df_filtrado[df_filtrado['Capacitado']]
    empleados_capacitados = empleados_capacitados[['ID Usuario', 'Centro de Trabajo']].drop_duplicates()

    # Total empleados por centro
    total_por_centro = (
        base_empleados.groupby('Centro de Trabajo')['ID Usuario']
        .nunique()
        .reset_index(name='Total Empleados')
    )

    # Empleados capacitados por centro
    cap_por_centro = (
        empleados_capacitados.groupby('Centro de Trabajo')['ID Usuario']
        .nunique()
        .reset_index(name='Empleados Capacitados')
    )

    # Unir y calcular cobertura
    centros_participacion = total_por_centro.merge(cap_por_centro, on='Centro de Trabajo', how='left').fillna(0)
    centros_participacion['Empleados Capacitados'] = centros_participacion['Empleados Capacitados'].astype(int)
    centros_participacion['% Cobertura'] = (centros_participacion['Empleados Capacitados'] / centros_participacion['Total Empleados'] * 100).round(1)

    # Top 12 centros con menor cobertura
    top_menor = centros_participacion.nsmallest(12, '% Cobertura')

    # Formatear para mostrar bonito
    top_menor_display = top_menor.copy()
    top_menor_display['% Cobertura'] = top_menor_display['% Cobertura'].astype(str) + " %"
    top_menor_display['Empleados Capacitados'] = top_menor_display['Empleados Capacitados'].map('{:,}'.format)
    top_menor_display['Total Empleados'] = top_menor_display['Total Empleados'].map('{:,}'.format)

    # Mostrar tabla ajustada al ancho de la página
    st.table(top_menor_display)

#----------------------------------------------------------------------------------------------------------------------

else:  
    st.markdown("## 🌐 Análisis por Segmentación")
    st.markdown(
        "Esta sección muestra los resultados basados en los filtros aplicados "
        "desde la barra lateral."
    )
    
    # Total de empleados en la segmentación
    total_empleados = df_filtrado['ID Usuario'].nunique()
    st.markdown(f"Cantidad de empleados: {total_empleados:,}")

    total_empleados = df_filtrado['ID Usuario'].nunique()

    # --- Resumen Visual del Segmento ---
    col_torta, col_info = st.columns([1, 1])

    con_cap = df_filtrado[df_filtrado['Capacitado'] == True]['ID Usuario'].nunique()
    sin_cap = total_empleados - con_cap

    with col_torta:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [con_cap, sin_cap], 
            labels=['Con capacitación', 'Sin capacitación'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#FF8C00', '#FFCC99'],
            textprops={'weight':'bold', 'fontsize':10}
        )
        ax.axis('equal')
        st.pyplot(fig)
        plt.clf()

    with col_info:
        st.metric("Total Empleados", f"{total_empleados:,}")
        st.write(f"✅ **Con capacitación:** {con_cap:,}")
        st.write(f"⏳ **Sin capacitación:** {sin_cap:,}")

    st.divider() # Separador visual antes de los detalles por Training

    # -------------------- CONTAR POR CAPACITACIÓN Y ESTADO --------------------

    st.markdown(f"### 📊 Resultados de las Capacitaciones por Training")
    # Solo usamos df_filtrado; Capacitado es booleano
    estado_df = (
        df_filtrado
        .groupby(['Capacitación', 'Estado'])['ID Usuario']
        .nunique()
        .unstack(fill_value=0)
    )

    # Ordenar por total de usuarios
    if not estado_df.empty:
        estado_df = estado_df.loc[estado_df.sum(axis=1).sort_values(ascending=False).index]

    # Colores para cada estado
    colores = {
        "Pendiente": "#FFA07A",  # naranja claro
        "Aprobado": "#FF8C00",   # naranja fuerte
        "Reprobado": "#FF6347"   # rojo tomate
    }
    colores_lista = [colores.get(x, "#999999") for x in estado_df.columns]

    # Mostrar tabla
    st.dataframe(estado_df)

    # -------------------- COLABORADORES MÁS ACTIVOS --------------------
    st.markdown("### 👥 Colaboradores más activos")
    conteo_usuarios = (
        df_filtrado.groupby('Usuario')['Capacitado']
        .sum()
        .reset_index(name='Cantidad de Capacitaciones Tomadas')
        .sort_values('Cantidad de Capacitaciones Tomadas', ascending=False)
    )
    st.table(conteo_usuarios.head(10).style.set_properties(**{'text-align': 'center'}))

    # -------------------- TABLA DE RESULTADOS --------------------
    st.markdown("#### Detalle de resultados:")
    st.dataframe(
        df_filtrado[[
            "ID Usuario", "Usuario", "Fecha", "Capacitación", "Estado", "Capacitado"
        ]],
        use_container_width=True
    )
