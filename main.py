from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder="templates")

# load trained SVM model (make sure svm_model.pkl exists in this folder!)
svm_model = pickle.load(open(
    r"D:/PYTHON VSCODE/six-months_python_for_data_science-mentorship-program-main/15_flask_web_apps/00_Complete_tutorials/06_diabetese_prediction/svm_model.pkl", 
    "rb"   # <-- important
))


def std_scalar(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/send", methods=["POST"])
def getdata():
    # collect features from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])   # shape (1, n_features)

    # scale input
    feature_transform = std_scalar(final_features)

    # make prediction
    prediction = svm_model.predict(feature_transform)[0]
    result = "You Are Diabetic" if prediction == 1 else "You Are Non-Diabetic"

    return render_template(
        "show.html",
        preg=request.form["Pregnancies"],
        gluc=request.form["Glucose"],
        bp=request.form["BloodPressure"],
        st=request.form["SkinThickness"],
        ins=request.form["Insulin"],
        bmi=request.form["BMI"],
        dbf=request.form["DiabetesPedigreeFunction"],
        age=request.form["Age"],
        res=result,
    )

if __name__ == "__main__":
    app.run(debug=True)

