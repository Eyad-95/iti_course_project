import numpy as np
import pickle
import pandas as pd
import streamlit as st
import csv
import matplotlib.pyplot as plt

# importing model from pickle file
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)
model = classifier["model"]
cm = classifier["cm"]
# predict whether the body mass is hazardous or not


def predict_danger_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    X = df[['relative_velocity', 'miss_distance']]
    output = model.predict(X)
    return output


def predict_danger_single(arr):
    X = np.array([arr])
    output = model.predict(X)
    return output


def download_csv_file():
    with open('data.csv', 'r') as file:
        data = file.read()
    st.download_button(label="Download CSV File",
                       data=data, file_name='data.csv')


def input_validation(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def save_data_to_csv(item):
    # Open the CSV file in write mode
    with open('data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([item])


def create_confusion_matrix(true_labels, predicted_labels):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = np.unique(true_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white")
    plt.tight_layout()
    return plt


def main():
    # initial markup and initiation of important flags
    flag = 0

    st.title("Outer Space")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Is Death Upon Us?</h2>
    </div>
    <hr>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    result = ""

    # for only one input
    relative_velocity = st.text_input("Relative Velocity", "")
    distance = st.text_input("Distance", "")

    # importing dataset for test
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if st.button("Predict"):
        # Validation, checking if user populated input fields and uploaded dataset
        if (uploaded_file is None and (relative_velocity == "" and distance == "")):
            st.warning("Please choose an option")
        elif (uploaded_file is not None and (relative_velocity != "" or distance != "")):
            st.warning("Please choose only one option")
        elif uploaded_file is not None and (relative_velocity == "" or distance == ""):
            # if a file was uploaded, run the model, output to a csv file
            result = predict_danger_csv(uploaded_file)
            for i, item in enumerate(result):
                save_data_to_csv(item)
            st.success("Succss")
            flag = 1
        elif relative_velocity == "" or distance == "":
            st.warning("Please populate all fields")
        else:
            if input_validation(relative_velocity) and input_validation(distance):
                result = predict_danger_single(
                    [float(relative_velocity), float(distance)])
                st.write("Result is: &nbsp;&nbsp;&nbsp;**{}**".format(
                    "<span style='color: red;'>Hazardous</span>" if result[0] == True else "<span style='color: green;'>Not Hazardous</span>"), unsafe_allow_html=True)
                with st.expander("Show Confusion Matrix"):
                    st.pyplot(create_confusion_matrix(
                        "Actual", "Predicted"))
                st.success("Succss")
            else:
                st.warning("Values should be numbers")

    if flag == 1:
        with open('data.csv', 'r') as file:
            data = file.read()
        st.download_button(label="Download CSV File",
                           data=data, file_name='data.csv')

    with st.expander("Instructions"):
        st.write("It is pretty simple. You have two options:")
        st.markdown(
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 - Enter only one entry. Just type in the relative velocity and distance of the mass body.")
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2 - Upload a csv file that includes both of these features titles "relative_velocity" and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"miss_distance".')
        st.markdown("Then, click Predict. And voila! You will get the output either in a form of a message, if option 1 was selected, or a csv file, if option 2 was selected.")


if __name__ == '__main__':
    main()
