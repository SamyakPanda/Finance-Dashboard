import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import nltk
from datetime import datetime
from streamlit_option_menu import option_menu
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cred

import requests

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["News","Stock Analysis","MonteCarlo Simulations"]
    )

if selected == "News":
    st.title('Stock Dashboard')
    st.write("Welcome to our Stock Dashboard, your comprehensive platform for tracking and analyzing stocks. Stay updated with real-time stock prices, financial data, and insightful charts to make informed investment decisions")
    def display_image_from_getty(link):
        st.image(link, use_column_width=True)

    headers = {
	"X-RapidAPI-Key": cred.token,
	"X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
    }
    url = "https://seeking-alpha.p.rapidapi.com/news/v2/list-trending"
    querystring = {"size":"20"}
    response = requests.get(url, headers=headers, params=querystring)
    trend_json = response.json()
    link_json = trend_json['data']
    count = 0
    
    st.header("Live Updates from the markets ")
    st.write("\n")
    for img in link_json:
        if(count<5):
            st.subheader(img['attributes']['title'])
            link = img['links']['uriImage']
            display_image_from_getty(link)
            st.write(img['links']['self'])
            st.write("\n")
        count = count + 1
if selected=="Stock Analysis":
    import requests
    default_date = datetime(2022, 6, 22)
    ticker = st.sidebar.text_input('Ticker', value="AAPL")
    start_date = st.sidebar.date_input('Start Date', value=default_date)
    end_date = st.sidebar.date_input('End Date')
    tickerData = yf.Ticker(ticker)
    website = tickerData.info['website']

    import requests

    import json
    url = "https://seeking-alpha.p.rapidapi.com/symbols/get-summary"

    querystring = {"symbols":ticker}

    headers = {
        "X-RapidAPI-Key": cred.token,
        "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
    }

    alpha_response = requests.get(url, headers=headers, params=querystring)

    alpha_json = alpha_response.json()

    info_alpha = alpha_json['data']

    for ap in info_alpha:
        
        st.subheader(ap['id'])
        st.write("Financials")

        st.write("PE Ratio:",ap['attributes']['peRatioFwd'])
        st.write("52 Week High:",ap['attributes']['high52'])
        st.write("52 Week Low:",ap['attributes']['low52'])

    data = yf.download(ticker,start=start_date,end=end_date)
    fig = px.line(data, x = data.index, y = data['Adj Close'])
    st.plotly_chart(fig)

    from prophet import Prophet
    from prophet.plot import plot_plotly
    from plotly import graph_objs as go

    st.title('Forecasting')
    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365

    def load_data(ticker):
        data = yf.download(ticker,start=start_date,end=end_date)
        data.reset_index(inplace=True)
        return data

    data = load_data(ticker)

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    symbol = ticker
    start_d = "2022-01-01"
    end_d = datetime.now().strftime("%Y-%m-%d")
    stock_data = yf.download(ticker,start=start_d,end=end_d)
    closing_price = stock_data['Close'][0]
    st.write("\n")
    st.header("News Stand")
    url = "https://mboum-finance.p.rapidapi.com/ne/news/"
    querystring = {"symbol":ticker}

    headers = {
        "X-RapidAPI-Key": cred.token,
        "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data_json = response.json()

    info_json = data_json['item']


    count = 0
    for news in info_json:
        if(count<3):
            st.subheader(news['title'])
            st.write(news['pubDate'])
            st.write(news['description'])
            st.write(news['link'])
        count = count + 1
    



if selected=="MonteCarlo Simulations":
    import matplotlib.pyplot as plt
    st.title('Monte Carlo Simulation')
    start_d = "2022-01-01"
    end_d = datetime.now().strftime("%Y-%m-%d")
    ticker = st.sidebar.text_input('Stock', value="AAPL")
    stock_data = yf.download(ticker,start=start_d,end=end_d)
    
    volatility = st.sidebar.text_input('Volatility', value="0.2")
    timesteps = st.sidebar.text_input('Days (Timesteps)', value="30")
    # timesteps = st.sidebar.text_input('Days (Timesteps)', value="30"

    # timesteps=30
    starting_price=stock_data['Close'][0]
    # volatility=0.02
    max_simulation=5000

    random_montecarlo=0.02*np.random.normal(0,1,30)
    random_montecarlo= np.insert(random_montecarlo, 0, 0)
    random_montecarlo= np.insert(random_montecarlo, 0, 0)

    a=np.zeros((max_simulation,31))
    for i in range(max_simulation):
        random_montecarlo=0.02*np.random.normal(0,1,30)
        random_montecarlo= np.insert(random_montecarlo, 0, 0)
        price_path=starting_price*(random_montecarlo+1).cumprod()
        price_path.reshape((1,31))    
        # plt.plot(price_path)
        for j in range(31):
            a[i,j]=price_path[j]
    b=a

    b=np.delete(b,0,axis=1)
    fig, ax = plt.subplots()
    ax.hist(b, bins=20)

    st.pyplot(fig)


