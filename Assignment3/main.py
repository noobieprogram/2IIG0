import pandas as pd
from sklearn.preprocessing import normalize, minmax_scale
from HiddenLayer import HiddenLayer, Neuron

df = pd.read_excel("HW3train.xlsx").drop(['y'], axis=1)
scaled = minmax_scale((df.as_matrix()))
print(scaled[17])

h1 = HiddenLayer(scaled[143])
h1.set_activation_values()
print(h1.activation_values)

h2 = HiddenLayer(h1.activation_values)
h2.set_activation_values()
print(h2.activation_values)

final = Neuron(h2.activation_values, "sigmoid")
print(final.activation_value)
