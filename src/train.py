import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("IrisClassifier")

with mlflow.start_run():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
