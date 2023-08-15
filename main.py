import numpy as np
import pickle
import pandas as pd
import streamlit as st
import csv

# importing model from pickle file
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# predict whether the body mass is hazardous or not


def predict_danger_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    X = df[["registered", "casual"]]
    output = classifier.predict(X)
    return output


def predict_danger_single(arr):
    X = np.array([arr])
    output = classifier.predict(X)
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
    # initial markup and initiation of important flags
    flag = 0

    st.title("Outer Space")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Is Death Upon Us?</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    result = ""

    # for only one input
    opt1 = st.text_input("opt1", "")
    opt2 = st.text_input("opt2", "")
    # opt3 = st.text_input("opt3", "")
    # opt4 = st.text_input("opt4", "")

    # importing dataset for test
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if st.button("Predict"):
        # Validation, checking if user populated input fields and uploaded dataset
        if (uploaded_file is not None and (opt1 != "" and opt2 != "")):
            st.text("Please choose only one option")
        elif uploaded_file is not None:
            # if a file was uploaded, run the model, output to a csv file
            result = predict_danger_csv(uploaded_file)
            for i, item in enumerate(result):
                save_data_to_csv(item)
            # with st.expander("See Predictions"):
            #     for i, item in enumerate(result):
            #         st.write(i+1, " {}".format(round(item, 2)))
            #         save_data_to_csv(item)
            st.success("Succss")
            flag = 1
        else:
            result = predict_danger_single([int(opt1), int(opt2)])
            st.write("Result is: {}".format(round(result[0])))
            st.success("Succss")

    if flag == 1:
        with open('data.csv', 'r') as file:
            data = file.read()
        st.download_button(label="Download CSV File",
                           data=data, file_name='data.csv')


if __name__ == '__main__':
    main()
