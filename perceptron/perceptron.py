def perceptron(x1, x2, w1, w2, bias):
    result = x1*w1 + x2*w2 + bias

    return result

def activationFunction(value):
    if value >= 0:
        return 1
    else: 
        return 0

def prediction(x1, x2, w1, w2, bias):
    result = perceptron(x1, x2, w1, w2, bias)

    value = activationFunction(result)

    return value

def updateWeights(w1, w2, bias, x1, x2, erro, learningRate):
    newW1 = w1 + learningRate * erro * x1
    newW2 = w2 + learningRate * erro * x2

    newBias = bias + learningRate * erro

    returnList = [newW1, newW2, newBias]

    return returnList


def train(data, labels, w1, w2, bias, learningRate, epochs):
    i = 0

    itens = [w1, w2, bias]

    while i < epochs:
        i2 = 0
        while i2 < len(data):
        
            predict = prediction(data[i2][0], data[i2][1], itens[0], itens[1], itens[2])

            erro = labels[i2] - predict

            itens = updateWeights(itens[0], itens[1], itens[2], data[i2][0], data[i2][1], erro, learningRate)

            i2 += 1
        i +=1

    return itens

def evaluate(data, labels, w1, w2, bias):
    predictList = []
    right = 0

    for item in data:
        predictList.append(prediction(item[0], item[1], w1, w2, bias))

    i= 0 
    while i < len(labels):
        if predictList[i] == labels[i]:
            right += 1
        i += 1

    accuracy = right / len(labels) * 100

    return accuracy

data = [
    (1, 1),
    (1, 2),
    (2, 1),
    (4, 4),
    (5, 4),
    (4, 5)
]

labels = [0, 0, 0, 1, 1, 1]

labels = [0, 0, 0, 1, 1, 1]

trainedWeights = train(data, labels, 0, 0, 0, 0.1, 5)
print(trainedWeights)
print(evaluate(data, labels, trainedWeights[0], trainedWeights[1], trainedWeights[2]))
