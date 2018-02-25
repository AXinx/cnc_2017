# coding=utf-8

import tensorflow as tf
import numpy as np
import pickle
import random
import os
import matplotlib.pyplot as plt

# defines the size of the batch.
BATCH_SIZE = 100
# one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42223

IMAGE_SIZE = 48

NUM_LABELS = 2

data_path = './data'

file1 = os.path.join(data_path,'BA_train.pkl')
file2 = os.path.join(data_path,'WS_train.pkl')
file3 = os.path.join(data_path,'BA_test.pkl')
file4 = os.path.join(data_path,'WS_test.pkl')

# prepare train datas and labels
f1 = open(file1,'rb')
content1 = pickle.load(f1,encoding='iso-8859-1')
f2 = open(file2,'rb')
content2 = pickle.load(f2,encoding='iso-8859-1')
dummy_train_data = content1[:4000] + content2[:4000]

dummy_train_labels = np.zeros((8000,2))
dummy_train_labels[:4000, 0 ] = 1
dummy_train_labels[4000:, 1 ] = 1

data_label_pair = list(zip(dummy_train_data, dummy_train_labels))
random.shuffle(data_label_pair)

train_data_temp = list(zip(*data_label_pair))[0]
train_labels_temp = list(zip(*data_label_pair))[1]

train_data = np.array(train_data_temp).reshape((8000,48,48,1)).astype(np.float32)
train_labels = np.array(train_labels_temp)

train_size = train_labels.shape[0]

#prepare validation data and labels
dummy_val_data = content1[4000:] + content2[4000:]

dummy_val_labels = np.zeros((2000,2))
dummy_val_labels[:1000, 0 ] = 1
dummy_val_labels[1000:, 1 ] = 1

val_label_pair = list(zip(dummy_val_data, dummy_val_labels))
random.shuffle(val_label_pair)

val_data_temp = list(zip(*val_label_pair))[0]
val_labels_temp = list(zip(*val_label_pair))[1]

val_data = np.array(val_data_temp).reshape((2000,48,48,1)).astype(np.float32)
val_labels = np.array(val_labels_temp)

val_size = val_labels.shape[0]

# prepare test datas and labels
f3 = open(file3,'rb')
content3 = pickle.load(f3,encoding='iso-8859-1')
f4 = open(file4,'rb')
content4 = pickle.load(f4,encoding='iso-8859-1')
dummy_test_data = content3 + content4

dummy_test_labels = np.zeros((1200,2))
dummy_test_labels[:600, 0 ] = 1
dummy_test_labels[600:, 1 ] = 1

test_data_label_pair = list(zip(dummy_test_data, dummy_test_labels))
random.shuffle(test_data_label_pair)

test_data_temp = list(zip(*test_data_label_pair))[0]
test_labels_temp = list(zip(*test_data_label_pair))[1]

test_data = np.array(test_data_temp).reshape((1200,48,48,1)).astype(np.float32)
test_labels = np.array(test_labels_temp)

# training
train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, NUM_LABELS))

val_data_node = tf.constant(val_data)
test_data_node = tf.constant(test_data)

# parameter initialize.
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 3],  # 5x5 filter
                      stddev=0.1,
                      seed=SEED))
conv1_biases = tf.Variable(tf.zeros([3]))
conv2_weights = tf.Variable(
  tf.truncated_normal([5, 5, 3, 5],
                      stddev=0.1,
                      seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[5]))
fc1_weights = tf.Variable(  # fully connected, depth 50.
  tf.truncated_normal([int(IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * 5), 50],
                      stddev=0.1,
                      seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[50]))
fc2_weights = tf.Variable(
  tf.truncated_normal([50, NUM_LABELS],
                      stddev=0.1,
                      seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

def model(data, train=False):
    """The Model definition."""
    # shape matches the data layout: [image index, y, x, depth].
    conv1 = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    pool1 = tf.nn.max_pool(relu1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
    conv2 = tf.nn.conv2d(pool1,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

    # fully connected layers.
    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
      pool2,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # 50% dropout
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    return error

logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  logits=logits, labels=train_labels_node))

# L2 regularization
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

loss += 5e-4 * regularizers

batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

optimizer = tf.train.MomentumOptimizer(learning_rate,
                                       0.9).minimize(loss,
                                                     global_step=batch)

train_prediction = tf.nn.softmax(logits)
val_prediction = tf.nn.softmax(model(val_data_node))
test_prediction = tf.nn.softmax(model(test_data_node))

#train
s = tf.InteractiveSession()
#save model
saver = tf.train.Saver()

tf.add_to_collection('x', train_data_node)
tf.add_to_collection('y', train_prediction)

s.as_default()

tf.initialize_all_variables().run()

steps = int(train_size / BATCH_SIZE)

los = []
st = []
train_err = []
val_err = []
for step in range(steps):
    print(step)
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    print(offset)
    batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

    feed_dict = {train_data_node: batch_data,
                train_labels_node: batch_labels}
    _, l, lr, predictions = s.run(
      [optimizer, loss, learning_rate, train_prediction],
      feed_dict=feed_dict)
    st.append(step)
    los.append(l)
    error_t = error_rate(predictions, batch_labels)
    error_v = error_rate(val_prediction.eval(),val_labels)
    train_err.append(error_t/100)
    val_err.append(error_v/100)
    print('Step %d of %d' % (step, steps))
    print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error_t, lr))
    print('validation error: %.1f%%' % error_v)

    if (step+1)%80 == 0:
        saver.save(s,'model_1', global_step=step)
'''
plt.figure(1)
plt.title('Loss curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(st,los)
plt.show()
plt.savefig('loss_curve')

plt.figure(2)
plt.title('Train error vs validation error',fontsize=15)
plt.xlabel('Iteration',fontsize=12)
plt.ylabel('Error rate',fontsize=12)
plt.plot(st,train_err, label='train error')
plt.plot(st,val_err, label='val error')
plt.legend(loc = 'upper right')
plt.savefig('error_curve')
plt.savefig('error_curve.eps')
'''
size1 = 15
size2 = 12
fig = plt.figure()
plt.title('Loss and Error', fontsize = size1)
ax1 = fig.add_subplot(1,1,1)
l1 = ax1.plot(st, los, 'r', label='loss')
#plt.legend(bbox_to_anchor=(1.0,0.15))
ax1.set_ylabel('Loss',fontsize = size2)
ax2 = ax1.twinx()
l2 = ax2.plot(st, val_err, 'g', label='error')
ls = l1+l2
labs = [l.get_label() for l in ls]
ax1.legend(ls, labs, bbox_to_anchor=(1.0,0.95))
ax2.set_ylabel('Error rate',fontsize = size2)
ax1.set_xlabel('Iteration',fontsize = size2)
plt.savefig('loss_err.png')
plt.savefig('loss_err.eps')
plt.show()

# Store variable
W_conv1 = conv1_weights.eval(s)
b_conv1 = conv1_biases.eval(s)
W_conv2 = conv2_weights.eval(s)
b_conv2 = conv2_biases.eval(s)
W_fc1 = fc1_weights.eval(s)
b_fc1 = fc1_biases.eval(s)
W_fc2 = fc2_weights.eval(s)
b_fc2 = fc2_biases.eval(s)

output1 = open('w_conv1.pkl','wb')
pickle.dump(W_conv1,output1)
output1.close()
output2 = open('w_conv2.pkl','wb')
pickle.dump(W_conv2,output2)
output2.close()

y_score = test_prediction.eval()
test_error = error_rate(y_score, test_labels)
print(y_score)
print(test_labels)
print('Test error: %.1f%%' % test_error)

'''
#roc = metrics.roc_auc_score(test_labels,test_prediction.eval())
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw = 2
plt.figure(2)
colors = cycle(['aqua', 'darkorange'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})\'''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class')
plt.legend(loc="lower right")
plt.show()
'''