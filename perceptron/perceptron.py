import numpy as np


TOLERANCE = 1e-10


def calculateWeightedSum(features, weights, bias):
    result = np.dot(features, weights) + bias

    return result


def activationFunction(value):
    if value >= 0:
        return 1
    else:
        return 0


def prediction(features, weights, bias):
    result = calculateWeightedSum(features, weights, bias)

    return activationFunction(result)


def updateWeights(weights, bias, features, error, learningRate):
    newWeights = weights + learningRate * error * np.array(features)
    newBias = bias + learningRate * error

    return newWeights, newBias


def train(data, labels, learningRate, epochs):
    weights = np.zeros(len(data[0]))
    bias = 0

    for _ in range(epochs):
        for features, label in zip(data, labels):

            predict = prediction(features, weights, bias)

            error = label - predict

            weights, bias = updateWeights(
                weights,
                bias,
                features,
                error,
                learningRate
            )

    return weights, bias


def evaluate(data, labels, weights, bias):
    correct = 0

    for features, label in zip(data, labels):
        predict = prediction(features, weights, bias)

        if predict == label:
            correct += 1

    accuracy = correct / len(labels) * 100

    return accuracy
