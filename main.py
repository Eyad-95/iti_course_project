import numpy as np
import pickle
import pandas as pd
import streamlit as st
import csv
import matplotlib.pyplot as plt

# importing model from pickle file
pickle_in = open("classifier.pkl", "rb")
x_tst_mean = 18569582.011831895
x_test_std = 24353612.333239723
classifier = pickle.load(pickle_in)
model = classifier["model"]
cm = classifier["cm"]
scaler = classifier["scaler"]
# predict whether the body mass is hazardous or not


def predict_danger_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    X = df[['relative_velocity', 'miss_distance']]
    X = scaler.transform(X)
    output = model.predict(X)
    dff = pd.DataFrame({'Hazardous': output})
    output_filename = 'model_predictions.csv'
    # index=False to exclude row indices in CSV
    dff.to_csv(output_filename, index=False)


def predict_danger_single(arr):
    X = scaler.transform([arr])
    output = model.predict(X)
    return output


def input_validation(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


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
    relative_velocity = st.text_input("Relative Velocity, in Kmph", "")
    distance = st.text_input("Distance, in Kilometers", "")

    # importing dataset for test
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if st.button("Predict"):
        # Validation, checking if user populated input fields and uploaded dataset
        if (uploaded_file is None and (relative_velocity == "" and distance == "")):
            st.warning("Please choose an option")
        elif (uploaded_file is not None and (relative_velocity != "" or distance != "")):
            st.warning("Please choose only one option")
        elif uploaded_file is not None and (relative_velocity == "" or distance == ""):
            predict_danger_csv(uploaded_file)
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
        with open('model_predictions.csv', 'r') as file:
            data = file.read()
        st.download_button(label="Download CSV File",
                           data=data, file_name='model_predictions.csv')

    with st.expander("Documentation"):

        # Model description
        st.header("Model Description")
        st.write(
            "The Outer Space Prediction Model is a supervised machine learning model that takes the relative velocity and distance of a body mass in outer space"
            "and predicts whether it is hazardous or not based on a Logistic Regression algorithm."
        )

        # Usage section
        st.header("Usage")

        # Installation
        st.subheader("Installation")
        st.code("pip install scikit-learn numpy")
        st.write("Install dependencies in the requirements.txt file")
        st.code("pip install -r requirements.txt")
        st.write("or install them one by one using **pip install**")

        st.write("**requirements.txt**")
        st.code('''streamlit==1.3.1
pandas==1.3.5
numpy==1.21.5
sklearn==0.0
scikit-learn==1.0.2
matplotlib==3.5.1
''')

        # Importing the model
        st.subheader("Importing the Model")
        st.code("from sklearn.linear_model import LogisticRegression")

        # Creating an instance
        st.subheader("Creating an Instance")
        st.code("model = LogisticRegression()")

        # Making predictions
        st.subheader("Making Predictions")
        st.code(
            """
        relative_velocity = 13569.2492241812
        distance = 54839744.082846
        prediction = model.predict(relative_velocity, distance)
        print('Result is:', prediction)
            """
        )

        # Inputs and Outputs sections
        st.header("Inputs and Outputs")

        # Inputs
        st.subheader("Inputs")
        st.write(
            "The model takes the following inputs:\n"
            "- **Relative Velocity**: Velocity of mass body in kmph.\n"
            "- **Distance**: Distance of mass body from Earth in km."
        )

        # Outputs
        st.subheader("Outputs")
        st.write(
            "The model produces the following output:\n"
            "- **Predicted Output**: Whether the mass body is hazardous to Earth or not."
        )

        # Example section
        st.header("Example")

        # Example code
        st.subheader("Example Code")
        st.code(
            """
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        relative_velocity = 13569.2492241812
        distance = 54839744.082846
        prediction = model.predict(relative_velocity, distance)

        print('Result is:', prediction)
            """
        )

        st.header("UI Usage")
        st.write("It is pretty simple. You have two options:")

        st.subheader("Option 1")
        st.write(
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Enter only one entry. Just type in the `relative velocity` and `distance` of the mass body.")
        st.subheader("Option 2")
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Upload a csv file that includes both of these features titles `relative_velocity` and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`miss_distance`.')

        st.markdown("Then, click `Predict`. And voila! You will get the output either in a form of a message, if `Option 1` was selected, or a csv file, if `Option 2` was selected.")
        # Notes section
        st.header("Notes")
        st.write(
            "- This example assumes you have a model class `LogisticRegression` with methods for prediction.\n"
            "- Replace the example class and methods with your actual model implementation.\n"
            "- Ensure you have the required dependencies installed.\n"
            "- Always preprocess input data appropriately before using the model for prediction."
        )

        # Add a footer
        st.markdown("---")
    with st.expander("Creators"):
        st.write("Eyad Magdy Gaber Khedr Hussein")
        st.write("Mahmoud Akram Mahmoud Mohamed")
        st.write("Toqa Alaa Abdel Rasoul Awad")
        st.write("Nada Ashraf Moussa Kamel")
        st.write("Ziad Hesham Al-Safy Rohayiem")
        st.write("Steven Adel Ata Yakoub")


if __name__ == '__main__':
    main()
