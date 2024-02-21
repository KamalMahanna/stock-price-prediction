from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import pandas_market_calendars as mcal
from datetime import timedelta



# give the next businessday
def next_business_day(today):
    nse = mcal.get_calendar('NSE')
    return nse.valid_days(start_date= today+timedelta(1), end_date=today+timedelta(7))[0]


model = load_model("./univariate/saved_model")
scaler = joblib.load("./univariate/scaler.pkl")
df=yf.download(tickers='GOOG',period="30d")['Close']

# scaled data, so as to predict by the model
reshaped_values=scaler.transform(df.values.reshape(-1,1)).reshape(1,30,1)
# the prediction
predicted_values= scaler.inverse_transform(model.predict(reshaped_values))[0][0]



# the web app
st.title("Google's Stock price prediction")

st.markdown(f'Market will be closed with a price: $<span style="color: red;">{predicted_values:4f}</span>',unsafe_allow_html=True )

# figure 

# st.dataframe(df)

fig = go.Figure()

fig.add_scatter(x=df.index,y=df.values,name="Previous")
fig.add_scatter(x=[next_business_day(df.index[-1])], 
                y=[predicted_values],
                mode='markers',
                name='Predicted',
                marker=dict(color='#ff1100' if predicted_values<df.values[-1] else '#09ff00'))

fig.update_layout(hovermode='x unified')
st.plotly_chart(fig)

# st.write(df.index[-1])
# st.write(next_business_day(df.index[-1]))