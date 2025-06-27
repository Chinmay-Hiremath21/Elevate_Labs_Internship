This task builds a **binary classification model** using Logistic Regression to predict whether a rental listing is **high rent (≥ ₹15,000)** or not.

1. Load the cleaned dataset (`House_Rent_Dataset_Cleaned.csv`). <br>
2. Create a new binary column `High_Rent` (1 if Rent ≥ 15,000, else 0).<br>
3. Select features: `Size`, `Bathroom`, `Floor`.<br>
4. Split data into training and testing sets (80/20).<br>
5. Train a **Logistic Regression** model.<br>
6. Predict and evaluate using:<br>
   - Confusion Matrix  <br>
   - Precision, Recall, F1-Score  <br>
   - ROC-AUC Score & ROC Curve<br>
7. Plot the **Sigmoid Function** to explain probability output.<br>
