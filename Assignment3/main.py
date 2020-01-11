import pandas as pd
from sklearn.preprocessing import minmax_scale
import warnings

warnings.filterwarnings("ignore")

from Assignment3.HiddenLayer import HiddenLayer, Neuron

# shitty forward feed

df = pd.read_excel("HW3train.xlsx")
scaled = minmax_scale(df.drop(['y'], axis=1).as_matrix())
y = df['y'].tolist()

inputs = [scaled[2], scaled[167], scaled[85]]

for data in inputs:
    h1 = HiddenLayer(data)
    h1.set_activation_values()
    print(h1.activation_values)

    h2 = HiddenLayer(h1.activation_values)
    h2.set_activation_values()
    print(h2.activation_values)

    # one output neuron for now
    # choose some arbitrary value as the threshold between class 0 and 1
    output = Neuron(h2.activation_values, "sigmoid")
    print(output.activation_value)


def backprop():
    pass
