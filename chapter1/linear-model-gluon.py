
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

if __name__ == "__main__":

    # prepare data
    num_inputs = 2
    num_examples = 1000

    true_w = [2, -3.4]
    true_b = 4.2

    X = nd.random_normal(shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += .01 * nd.random_normal(shape=y.shape)


    batch_size = 10
    dataset = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)



    net = gluon.nn.Sequential()

    # dense(1) means this layer has only one neuron
    # and you need not to tell the model the demension of input
    net.add(gluon.nn.Dense(1))

    net.initialize()

    square_loss = gluon.loss.L2Loss()

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.01})



    epochs = 5
    batch_size = 10
    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

    dense = net[0]
    true_w, dense.weight.data()

    true_b, dense.bias.data()