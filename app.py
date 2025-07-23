import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Análisis de Series Temporales COVID-19", layout="wide")

# --- ESTILO PERSONALIZADO ---
st.markdown("""
    <style>
        .stApp {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        header { 
            visibility: hidden;
        }
        .block-container {
            padding-top: 1rem;
        }
        h1 {
            margin-top: -2rem;
        }
        h2 {
            font-size: 16px !important;
            color: red !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        h3, h4, h5, h6 {
            color: black;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        .subheader {
            color: #333;
        }
        .menu-button {
            background-color: #004080;
            color: white;
            padding: 10px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            border: none;
        }
        .menu-button:hover {
            background-color: #0059b3;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO GENERAL ---
st.title("**Análisis de Series Temporales - Datos COVID-19**")

# --- CARGAR DATOS ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(url)
    
    # Transformar datos de ancho a largo
    df = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                 var_name='Date', 
                 value_name='Confirmed')
    
    # Convertir fecha y ordenar
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Country/Region', 'Date'])
    
    # Calcular nuevos casos diarios
    df['Daily_Cases'] = df.groupby('Country/Region')['Confirmed'].diff().fillna(0)
    
    return df

df = load_data()

# --- BARRA LATERAL ---
st.sidebar.header("Parámetros de Análisis")

# Selección de país
countries = sorted(df['Country/Region'].unique())
selected_country = st.sidebar.selectbox("Seleccione un país", countries, index=countries.index('Spain'))

# Rango de fechas
min_date = df['Date'].min().to_pydatetime()
max_date = df['Date'].max().to_pydatetime()
start_date = st.sidebar.date_input("Fecha de inicio", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Fecha de fin", max_date, min_value=min_date, max_value=max_date)

# Convertir a datetime64
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Tipo de gráfico
chart_type = st.sidebar.radio("Tipo de visualización", 
                             ['Serie Completa', 'Promedio Móvil', 'Comparación de Periodos'])

# Parámetros específicos
if chart_type == 'Promedio Móvil':
    window_size = st.sidebar.slider("Ventana para promedio móvil (días)", 3, 30, 7)
elif chart_type == 'Comparación de Periodos':
    period1_start = st.sidebar.date_input("Inicio Periodo 1", datetime(2020, 3, 1), min_value=min_date, max_value=max_date)
    period1_end = st.sidebar.date_input("Fin Periodo 1", datetime(2020, 6, 30), min_value=min_date, max_value=max_date)
    period2_start = st.sidebar.date_input("Inicio Periodo 2", datetime(2020, 9, 1), min_value=min_date, max_value=max_date)
    period2_end = st.sidebar.date_input("Fin Periodo 2", datetime(2020, 12, 31), min_value=min_date, max_value=max_date)

# Modelo de predicción
model_type = st.sidebar.selectbox("Modelo de predicción", 
                                 ['Ninguno', 'Suavizado Exponencial Simple', 'Holt', 'Prophet'])

if model_type != 'Ninguno':
    forecast_days = st.sidebar.slider("Días a predecir", 7, 90, 30)

# --- FILTRAR DATOS ---
country_data = df[(df['Country/Region'] == selected_country) & 
                 (df['Date'] >= start_date) & 
                 (df['Date'] <= end_date)]

if country_data.empty:
    st.warning("No hay datos disponibles para los parámetros seleccionados.")
    st.stop()

# --- VISUALIZACIONES ---
st.header(f"Datos de COVID-19 para {selected_country}")

# Gráfico según selección
fig, ax = plt.subplots(figsize=(12, 6))

if chart_type == 'Serie Completa':
    ax.plot(country_data['Date'], country_data['Daily_Cases'], label='Casos diarios')
    ax.set_title(f'Casos diarios de COVID-19 en {selected_country}')
    
elif chart_type == 'Promedio Móvil':
    rolling_mean = country_data['Daily_Cases'].rolling(window=window_size).mean()
    ax.plot(country_data['Date'], country_data['Daily_Cases'], label='Casos diarios', alpha=0.3)
    ax.plot(country_data['Date'], rolling_mean, label=f'Promedio móvil {window_size} días', color='orange')
    ax.set_title(f'Casos diarios y promedio móvil en {selected_country}')
    
elif chart_type == 'Comparación de Periodos':
    period1 = country_data[(country_data['Date'] >= pd.to_datetime(period1_start)) & 
                          (country_data['Date'] <= pd.to_datetime(period1_end))]
    period2 = country_data[(country_data['Date'] >= pd.to_datetime(period2_start)) & 
                          (country_data['Date'] <= pd.to_datetime(period2_end))]
    
    ax.plot(period1['Date'], period1['Daily_Cases'], label=f'Periodo 1: {period1_start} a {period1_end}')
    ax.plot(period2['Date'], period2['Daily_Cases'], label=f'Periodo 2: {period2_start} a {period2_end}')
    ax.set_title(f'Comparación entre dos periodos en {selected_country}')

ax.set_xlabel('Fecha')
ax.set_ylabel('Número de casos')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- MODELO DE PREDICCIÓN ---
if model_type != 'Ninguno':
    st.header("Predicción de casos futuros")
    
    # Preparar datos para el modelo
    train_data = country_data[['Date', 'Daily_Cases']].rename(columns={'Date': 'ds', 'Daily_Cases': 'y'})
    
    if model_type == 'Suavizado Exponencial Simple':
        model = SimpleExpSmoothing(train_data['y']).fit()
        forecast = model.forecast(forecast_days)
        
    elif model_type == 'Holt':
        model = Holt(train_data['y']).fit()
        forecast = model.forecast(forecast_days)
        
    elif model_type == 'Prophet':
        prophet_model = Prophet(seasonality_mode='multiplicative')
        prophet_model.fit(train_data)
        future = prophet_model.make_future_dataframe(periods=forecast_days)
        forecast = prophet_model.predict(future)
    
    # Visualizar predicción
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    if model_type == 'Prophet':
        ax2.plot(train_data['ds'], train_data['y'], label='Datos reales')
        ax2.plot(forecast['ds'], forecast['yhat'], label='Predicción', linestyle='--')
        ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    else:
        last_date = train_data['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
        ax2.plot(train_data['ds'], train_data['y'], label='Datos reales')
        ax2.plot(future_dates, forecast, label='Predicción', linestyle='--')
    
    ax2.set_title(f'Predicción usando {model_type}')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Número de casos')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Mostrar métricas de error (solo para modelos no Prophet)
    if model_type != 'Prophet':
        # Usar los últimos 'n' días como test
        n_test = min(30, len(train_data) // 4)  # 25% de los datos o 30 días, lo que sea menor
        train = train_data[:-n_test]
        test = train_data[-n_test:]
        
        if model_type == 'Suavizado Exponencial Simple':
            model_train = SimpleExpSmoothing(train['y']).fit()
            pred = model_train.forecast(len(test))
        elif model_type == 'Holt':
            model_train = Holt(train['y']).fit()
            pred = model_train.forecast(len(test))
        
        # Calcular MAPE (línea corregida)
        mape = np.mean(np.abs((test['y'] - pred) / test['y'])) * 100
        st.write(f"Error de predicción (MAPE) en datos de prueba: {mape:.2f}%")

