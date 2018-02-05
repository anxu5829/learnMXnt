
# we use context to check if an array is running on gpu
import mxnet as mx
from mxnet import nd
x = nd.array([1,2,3])
x.context


# we use ctx to define the context of an array
a = nd.array([1,2,3], ctx=mx.gpu())
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
(a,b,c)


# you can output the error like this:
import sys

try:
    nd.array([1,2,3], ctx=mx.gpu(10))
except mx.MXNetError as err:
    sys.stderr.write(str(err))


# from cpu to gpu
# both method need the same parameters
y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())
(y, z)

# copyto will always copy the data
yy = y.as_in_context(mx.gpu())
zz = z.copyto(mx.gpu())
(yy is y, zz is z)



# when you build a net , you need to initialize it in gpu
from mxnet import gluon
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize(ctx=mx.gpu())


data = nd.random.uniform(shape=[3,2], ctx=mx.gpu())
net(data)

net[0].weight.data()