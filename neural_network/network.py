
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss):
        self.loss = loss

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result