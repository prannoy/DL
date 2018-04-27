import tensorflow as tf
import numpy as np
from scipy.misc import imsave as ims
from random import randint

#A simple GAN Network to generate images similar to CIFAR-10 Image Dataset

img_dim = 32 #CIFAR has 32x32 RGB Images
z_dim = 100
minibatch_size = 64 #minibatch size 
num_training_iterations = 1000

#Adam parameters suggested in https://arxiv.org/abs/1511.06434
learningrate = 0.0002
beta1=0.5

#Generator Weights
#G_W1 are the weights of the FC Layer which are handled automatically by TensorFlow
G_W2 = tf.get_variable("G_W2", [5, 5, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
G_W3 = tf.get_variable("G_W3", [5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
G_W4 = tf.get_variable("G_W4", [5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer())

def merge(images, size): #Used to merge images from a minibatch
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def unpickle(file): #For CIFAR-10 Dataset
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def discriminator(image, reuse=False):
    #Using the reverse of Generator Architecture as Discriminator Architecture

    A1 = tf.contrib.layers.conv2d(image, 64, kernel_size=5, padding="SAME", stride=[2,2], reuse=reuse, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.leaky_relu, normalizer_params = {"decay" : 0.95}, scope='D_A1')
    A2 = tf.contrib.layers.conv2d(A1, 128, kernel_size=5, padding="SAME", stride=[2,2], reuse=reuse, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.leaky_relu, normalizer_params = {"decay" : 0.95}, scope='D_A2')
    A3 = tf.contrib.layers.conv2d(A2, 256, kernel_size=5, padding="SAME", stride=[2,2], reuse=reuse, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.leaky_relu, normalizer_params = {"decay" : 0.95}, scope='D_A3')
    Z4 = tf.layers.dense(tf.reshape(A3, [minibatch_size, -1]), 4*4*256, reuse=reuse, name='D_Z4')
    return Z4
    

def generator(z):
    #Using a similar Generator Network Architecture described in DCGAN Publication (https://arxiv.org/abs/1511.06434)
    # z(z_dim x 1) (fully connected layer)-> z1(4*4*256 x 1) (reshape)-> h1(4 x 4 x 256) (Deconv, Stride 2)-> 
    # ->h2(8 x 8 x 128) (Deconv, Stride 2)-> h3(16 x 16 x 128) -> X_G(32 x 32 x 3)

    Z1  = tf.layers.dense(z, 4*4*256)
    A1  = tf.reshape(Z1, [-1, 4, 4, 256])
    A2  = tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(A1, G_W2, [minibatch_size, 8, 8, 128], strides=[1, 2, 2, 1]), activation_fn=tf.nn.relu, decay=0.95)
    A3  = tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(A2, G_W3, [minibatch_size, 16, 16, 64], strides=[1, 2, 2, 1]), activation_fn=tf.nn.relu, decay=0.95)
    X_G = tf.nn.tanh(tf.nn.conv2d_transpose(A3, G_W4, [minibatch_size, 32, 32, 3], strides=[1, 2, 2, 1]))
    return X_G

with tf.Session() as sess:
	images = tf.placeholder(tf.float32, [minibatch_size, 32, 32, 3])
	Zin = tf.placeholder(tf.float32, [minibatch_size, z_dim])
	G_images = generator(Zin)

	D_real_logit = discriminator(images)
	D_fake_logit = discriminator(G_images, reuse=True) #Won't update the Discriminator Network Weights while doing BackProp, only Generator weights are updated

	D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real_logit)))
	D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))

	D_loss = D_loss_real + D_loss_fake
	G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))
	#D_loss = tf.reduce_mean(-tf.log(D_real) - tf.log(1 - D_fake))
	#G_loss = tf.reduce_mean(-tf.log(1 - D_fake))

	training_images_batch = unpickle("cifar-10-batches-py/data_batch_2")

	training_vars = tf.trainable_variables()
	d_vars = [var for var in training_vars if 'D_' in var.name]
	g_vars = [var for var in training_vars if 'G_' in var.name]

	D_Optimizer = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(D_loss, var_list=d_vars)
	G_Optimizer = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(G_loss, var_list=g_vars)
	tf.initialize_all_variables().run()

	for i in range(num_training_iterations):
		total_minibatches = int(len(training_images_batch[b'data'])/minibatch_size)
		images_minibatch = None
		z_minibatch = None
		for minibatch_num in range(10*total_minibatches):
			minibatch_idx = randint(0, total_minibatches-1)
			training_minibatch = training_images_batch[b'data'][minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
			training_minibatch = np.array(training_minibatch)/127.5 - 1 #Activation Function of Generator is tanh to standardize real input into that range
			training_minibatch = np.reshape(training_minibatch,(minibatch_size,3,32,32))
			images_minibatch = np.swapaxes(training_minibatch,1,3)
			z_minibatch = np.random.uniform(-1, 1, [minibatch_size, z_dim]).astype(np.float32)

			sess.run([D_Optimizer], feed_dict={ images: images_minibatch, Zin: z_minibatch})
			sess.run([G_Optimizer], feed_dict={ Zin: z_minibatch})

		print ("Epoch Number ", 10*i, "completed")
		z_rand = np.random.uniform(-1, 1, [minibatch_size, z_dim]).astype(np.float32)
		sdata = sess.run([G_images], feed_dict={ Zin: z_rand})
		ims("gan_output_"+str(i)+".jpg",merge(sdata[0],[8,8]))
		D_cost = D_loss.eval({images: images_minibatch, Zin: z_minibatch})
		G_cost = G_loss.eval({Zin: z_minibatch})
		D_cost_real = D_loss_real.eval({images: images_minibatch})
		print ("Cost D ", D_cost, "D_real_loss ", D_cost_real)
		print ("Cost G ", G_cost)

     





