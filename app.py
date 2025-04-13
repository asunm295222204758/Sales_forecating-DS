import streamlit as st 
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import traceback

def create_forecast_plot(forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        mode='lines'
    ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Sales')
    return fig

def main():
    try:
        # Load your data
        df = pd.read_csv(r'C:\Users\egang\OneDrive\Desktop\Sales forecasting\Sales_forecating-DS\Walmart.csv')
        
        # Print column names to debug
        st.write("Original column names:", df.columns.tolist())
        
        # Clean column names by removing leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%d-%m-%Y')
        
        # Clean numeric columns only if they are strings
        numeric_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for col in numeric_columns:
            if df[col].dtype == 'object':  # Only clean if column contains strings
                df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Streamlit UI elements
        st.title('Sales Forecasting & Inventory Optimization')

        # User input: Store and Date Range
        store = st.selectbox('Select Store', sorted(df['Store'].unique()))
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        start_date = pd.to_datetime(st.date_input('Start Date', min_date))
        end_date = pd.to_datetime(st.date_input('End Date', max_date))

        # Filter the data based on user input
        store_data = df[df['Store'] == store].copy()
        
        # Prepare store-specific data for Prophet
        sales = store_data[['Date', 'Weekly_Sales']].rename(columns={
            'Date': 'ds',
            'Weekly_Sales': 'y'
        })
        
        # Prophet model for store-specific data
        model = Prophet()
        model.fit(sales)
        future = model.make_future_dataframe(periods=12, freq='W')
        forecast = model.predict(future)

        # Show forecast using plotly
        st.subheader(f"Sales Forecast for Store {store}")
        forecast_fig = create_forecast_plot(forecast, f"Sales Forecast - Store {store}")
        st.plotly_chart(forecast_fig)

        # Recommended Inventory
        forecast['recommended_stock'] = forecast['yhat'] * 1.10
        st.subheader(f"Recommended Inventory for Store {store}")
        inventory_fig = create_forecast_plot(
            forecast[['ds', 'recommended_stock']].rename(columns={'recommended_stock': 'yhat'}),
            f"Recommended Inventory - Store {store}"
        )
        st.plotly_chart(inventory_fig)

        # Show forecast table
        st.write("Forecast Details (Last 5 periods):")
        st.dataframe(forecast[['ds', 'yhat', 'recommended_stock']].tail())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__': 
    st.set_page_config(page_title="Sales Forecasting", layout="wide")
    main()
