from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluateKNN(k, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
     
    return accuracy

i = 1
while i <= 10:
    print(f"k = {i}: {evaluateKNN(i, X_train, y_train, X_test, y_test)}")
    i+=1




