from deeplearning_assistant.model_builder import AbstractModelBuilder

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge
		from keras.layers import concatenate
		from keras.layers.advanced_activations import LeakyReLU

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in xrange(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)

			h = Conv2D(2048, (1, 3), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Conv2D(2048, (1, 5), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Conv2D(2048, (1, 10), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Conv2D(2048, (1, 20), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Conv2D(2048, (1, 40), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

			h = Conv2D(2048, (1, 60), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

		m = concatenate(merges, axis = -1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(2, activation = 'softmax')(m)
		model = Model(output = V, inputs = inputs)

		return model

class MarketModelBuilder(AbstractModelBuilder):
	
	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization#, Merge
		from keras.layers import concatenate
		from keras.layers.advanced_activations import LeakyReLU

		dr_rate = 0.0

		B = Input(shape = (3,)) # <tf.Tensor 'input_1:0' shape=(?, 3) dtype=float32> # review: because the len(self.state[0])=3
		b = Dense(5, activation = "relu")(B) # <tf.Tensor 'dense_1/Relu:0' shape=(?, 5) dtype=float32>

		inputs = [B]
		merges = [b]
		merges1 = []

		for i in xrange(1):
			S = Input(shape=[4, 60, 1])
			inputs.append(S) # [<tf.Tensor 'input_1:0' shape=(?, 3) dtype=float32>, <tf.Tensor 'input_2:0' shape=(?, 2, 60, 1) dtype=float32>]

			h = Conv2D(64, (1,3), padding = "valid")(S) # <tf.Tensor 'conv2d_1/BiasAdd:0' shape=(?, 2, 58, 64) dtype=float32>
			h = LeakyReLU(0.001)(h) # <tf.Tensor 'leaky_re_lu_1/LeakyRelu/Maximum:0' shape=(?, 2, 58, 64) dtype=float32>
			h = MaxPooling2D(pool_size=(2,2), strides=None)(h) # review
			h = Flatten()(h) # review
			merges1.append(h) # review
			h = Conv2D(128, (1,5), padding = "valid")(S)#(S) # <tf.Tensor 'conv2d_2/BiasAdd:0' shape=(?, 2, 56, 128) dtype=float32>
			h = LeakyReLU(0.001)(h)
			h = MaxPooling2D(pool_size=(2, 2), strides=None)(h)  # review
			h = Flatten()(h) # review
			merges1.append(h)  # review
			h = Conv2D(256, (1, 10), padding = "valid")(S)#(S) # review: original
			h = LeakyReLU(0.001)(h) # review: original
			h = MaxPooling2D(pool_size=(2, 2), strides=None)(h)  # review
			h = Flatten()(h) # review
			merges1.append(h)  # review
			h = Conv2D(512, (1, 20), padding = "valid")(S)#(S) # review: original
			h = LeakyReLU(0.001)(h) # review: original
			h = MaxPooling2D(pool_size=(2, 2), strides=None)(h)  # review
			h = Flatten()(h) # review
			merges1.append(h)  # review
			h = Conv2D(1024, (1, 40), padding = "valid")(S)#(S) # review: original
			h = LeakyReLU(0.001)(h) # review: original
			h = MaxPooling2D(pool_size=(2, 2), strides=None)(h)  # review
			h = Flatten()(h) # review
			merges1.append(h)  # review
			m1 = concatenate(merges1,axis=-1) # review

			#h = Flatten()(m1) #(h) # review: original
			h = Dense(2048)(m1) #(h) # review: original
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

			h = Conv2D(2048, (1, 60), padding = "valid")(S)
			h = LeakyReLU(0.001)(h)
			#h = MaxPooling2D(pool_size=(2, 2), strides=None)(h)  # review

			h = Flatten()(h)
			h = Dense(4096)(h)
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

		m = concatenate(merges,axis = -1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		V = Dense(2, kernel_initializer = "zero", activation = "linear")(m)
		model = Model(outputs = V, inputs = inputs)

		return model
