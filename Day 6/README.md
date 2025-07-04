**What this code does (K-Nearest Neighbors Classification Overview)**<br><br>
- Step 1: Loads the cleaned Iris dataset from `cleaned_Iris.csv`.<br>
- Step 2: Selects relevant features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) to predict the target `Species`.<br>
- Step 3: Drops the `id` column as it's not useful for prediction.<br>
- Step 4: Splits the data into training and testing sets using an 80-20 ratio.<br>
- Step 5: Trains a K-Nearest Neighbors (KNN) classifier with `n_neighbors=3` using Scikit-learn.<br>
- Step 6: Makes predictions on the test set.<br>
- Step 7: Evaluates the model using accuracy score and confusion matrix.<br>
- Step 8: Visualizes the confusion matrix using Matplotlib.<br>
