from predict import predict_disease


# =========================================================
# HEART DISEASE TEST
# =========================================================

heart_input = [
    45,
    1,
    1,
    130,
    240,
    0,
    1,
    150,
    0,
    1.0,
    2,
    0,
    2
]


heart_result = predict_disease(
    "Heart Disease",
    heart_input
)

print("\n" + "=" * 60)
print("HEART DISEASE TEST")
print("=" * 60)

print("SVM           :", heart_result["SVM"])
print("KNN           :", heart_result["KNN"])
print("Random Forest :", heart_result["Random Forest"])
print("Ensemble      :", heart_result["Ensemble"])
print("Agreement     :", heart_result["Agreement"], "/ 3")
print("Final         :", heart_result["Final"])


# =========================================================
# DIABETES TEST
# =========================================================

diabetes_input = [
    2,
    140,
    80,
    25,
    100,
    32.0,
    0.5,
    35
]


diabetes_result = predict_disease(
    "Diabetes",
    diabetes_input
)

print("\n" + "=" * 60)
print("DIABETES TEST")
print("=" * 60)

print("SVM           :", diabetes_result["SVM"])
print("KNN           :", diabetes_result["KNN"])
print("Random Forest :", diabetes_result["Random Forest"])
print("Ensemble      :", diabetes_result["Ensemble"])
print("Agreement     :", diabetes_result["Agreement"], "/ 3")
print("Final         :", diabetes_result["Final"])