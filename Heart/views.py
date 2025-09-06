from django.shortcuts import render
import joblib
import numpy as np
import os

# Load ML model safely
model = None
model_load_error = None
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))
except Exception as e:
    model_load_error = str(e)

def heart_form(request):
    context = {"model_load_error": model_load_error, "invalid_fields": []}

    if request.method == "POST" and model is not None:
        values = request.POST
        invalid_fields = []

        try:
            # Collect and validate inputs
            age = int(values.get("age", "0"))
            if age < 1 or age > 120:
                invalid_fields.append("age")

            sex = int(values.get("sex", "-1"))
            if sex not in [0, 1]:
                invalid_fields.append("sex")

            cp = int(values.get("cp", "-1"))
            if cp not in [0, 1, 2, 3]:
                invalid_fields.append("cp")

            trestbps = int(values.get("trestbps", "0"))
            if trestbps < 50 or trestbps > 300:
                invalid_fields.append("trestbps")

            chol = int(values.get("chol", "0"))
            if chol < 100 or chol > 600:
                invalid_fields.append("chol")

            fbs = int(values.get("fbs", "-1"))
            if fbs not in [0, 1]:
                invalid_fields.append("fbs")

            restecg = int(values.get("restecg", "-1"))
            if restecg not in [0, 1, 2]:
                invalid_fields.append("restecg")

            thalach = int(values.get("thalach", "0"))
            if thalach < 30 or thalach > 250:
                invalid_fields.append("thalach")

            exang = int(values.get("exang", "-1"))
            if exang not in [0, 1]:
                invalid_fields.append("exang")

            oldpeak = float(values.get("oldpeak", "0"))
            if oldpeak < 0 or oldpeak > 10:
                invalid_fields.append("oldpeak")

            slope = int(values.get("slope", "-1"))
            if slope not in [0, 1, 2]:
                invalid_fields.append("slope")

            ca = int(values.get("ca", "-1"))
            if ca not in [0, 1, 2, 3, 4]:
                invalid_fields.append("ca")

            thal = int(values.get("thal", "-1"))
            if thal not in [0, 1, 2, 3]:
                invalid_fields.append("thal")

            if invalid_fields:
                # Send list of invalid fields to template
                context.update({
                    "values": values,
                    "invalid_fields": invalid_fields
                })
                return render(request, "heart_form.html", context)

            # Prepare input
            features = np.array([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak,
                                  slope, ca, thal]])

            # Predict
            prediction = model.predict(features)[0]
            result = "Likely" if prediction == 1 else "Not Likely"

            return render(request, "result.html", {
                "result": result,
                "values": values
            })

        except Exception as e:
            context.update({
                "values": values,
                "invalid_fields": [],  # treat as general system error
            })
            return render(request, "heart_form.html", context)

    return render(request, "heart_form.html", context)
