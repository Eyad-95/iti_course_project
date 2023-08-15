import numpy as np
import pickle
import pandas as pd
import streamlit as st
import csv

from PIL import Image

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(uploaded_file=None):
    # if opt1 == "":
    df = pd.read_csv(uploaded_file)
    X = df[["registered", "casual"]]
    output = classifier.predict(X)
    output_df = pd.DataFrame({'cnt': output})
    output_df.to_csv('linear_regression_output.csv', index=False)
    # else:
    # code for predicting single input
    return output


def download_csv_file():
    with open('data.csv', 'r') as file:
        data = file.read()
    st.download_button(label="Download CSV File",
                       data=data, file_name='data.csv')


def save_data_to_csv(item):
    # Open the CSV file in write mode
    with open('data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([item])


def main():
    flag = 0
    st.title("Project Deployment")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Project Deployment ITI</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    result = ""
    # for only one input
    opt1 = st.text_input("Enter your input", "")
    # importing dataset for test
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if st.button("Predict"):
        if (uploaded_file is not None and opt1 != ""):
            st.text("Please choose only one option")
        elif (uploaded_file is not None):
            result = predict_note_authentication(uploaded_file)
            with open('data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
            with st.expander("See Predictions"):
                for i, item in enumerate(result):
                    st.write(i+1, " {}".format(round(item, 2)))
                    save_data_to_csv(item)
            st.success("Succss")
            flag = 1
        else:
            result = predict_note_authentication(opt1)

    # if st.button("About"):
    #     st.text("Lets Learn")
    #     st.text("Built with Streamlit")
    # if st.button("Download CSV"):
    #     download_csv_file()

    if flag == 1:
        with open('data.csv', 'r') as file:
            data = file.read()
        st.download_button(label="Download CSV File",
                           data=data, file_name='data.csv')


if __name__ == '__main__':
    main()
