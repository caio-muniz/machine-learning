from perceptron import train, evaluate


data = [
    [1, 1],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 3],
    [3, 4]
]

labels = [0, 0, 0, 1, 1, 1]


weights, bias = train(
    data,
    labels,
    learningRate=0.1,
    epochs=10
)

print("Pesos:", weights)
print("Bias:", bias)

accuracy = evaluate(data, labels, weights, bias)

print("Accuracy:", accuracy)