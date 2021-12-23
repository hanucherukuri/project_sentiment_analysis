import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(message):
    model=load_model("best_lstm_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
        x_1 = tokenizer.texts_to_sequences([message])
        x_1 = pad_sequences(x_1, maxlen=400)
        predictions = model.predict(x_1)[0][0]
        return predictions

def main():

    st.title("Sentiment Analysis Web App")
    st.subheader("Enter the text you would like to analyse")
    message = st.text_area("Enter Text Here:")
    if st.button("Submit 👈"):
        with st.spinner("Analysing the text"):
            prediction=predict(message)
        if prediction > 0.6:
            st.subheader("Result")
            st.success("Positive sentiment 🙂 with {:2f} confidence".format(prediction*100))
            st.balloons()
        elif prediction <0.4:
            st.subheader("Result")
            st.error("Negative sentiment ☹️ with {:2f} confidence".format((1-prediction)*100))
    else:
        st.warning("Not sure 🤔! Try to add some more words")

if __name__ == '__main__':
    main()