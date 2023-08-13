from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import streamlit as st


model = load_model("./univariate/saved_model")
scaler = joblib.load("./univariate/scaler.pkl")
df=yf.download(tickers='GOOG',period="30d")['Close']

# scaled data, so as to predict by the model
reshaped_values=scaler.transform(df.values.reshape(-1,1)).reshape(1,30,1)
# the prediction
predicted_values= scaler.inverse_transform(model.predict(reshaped_values))[0][0]



# the web app
st.title("Google's Stock price prediction")

tab1, tab2 = st.tabs(["📈 Prediction","🤖 About the Model"])


with tab1:
    st.subheader(f'Market will be closed with a price: {predicted_values:4f}' )

with tab2:
    st.write("success")