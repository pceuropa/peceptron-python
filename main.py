from perceptron import Perceptron
# from data import logic_gate_and as train_data
from data import X_train, y
from data import validate_data
from ancillary import plot, plot_error_history


"""1.Initialization perceptron"""
perceptron = Perceptron(epochs=25, eta=0.01)

"""2.perceptron training"""
perceptron.train(X_train, target=y)
print(f'Weights: {perceptron.w}')

"""3.Prediction"""
prediction = perceptron.predict(X_train)
print(f'Output (Y): {prediction}')

print('last')
plot(X_train, prediction, perceptron, 25, None, perceptron.w)
plot_error_history(perceptron.err)
