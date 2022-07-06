import tensorflow as tf # import tensorflow
print(f"Tensorflow Version: {tf.__version__}") # print tensorflow version

mnist = tf.keras.datasets.mnist # load mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load mnist data
x_train, x_test = x_train / 255.0, x_test / 255.0 # convert data to floating point numbers

model = tf.keras.models.Sequential([ # create model through layering
	tf.keras.layers.Flatten(input_shape=(28, 28)), # flattens multi dimensional input into a single dimensional output
	tf.keras.layers.Dense(128, activation='relu'), # each neuron receives input from all the neurons of the prior layer
												   # relu, negatives go to 0
	tf.keras.layers.Dropout(0.2), # layer that prevents the model from overfitting
	tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy() # unnormalized
probabilities = tf.nn.softmax(predictions).numpy() # normalized/non linear

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # quantifies diff between expected outcome and outcome produced by the ml model
initial_loss = loss_fn(y_train[:1], predictions).numpy() # taking the expected outcome and comparing it to the coutcome of the model

model.compile(optimizer="adam", # compile model by selecting learning rate optimizer
			  loss=loss_fn,	    # evaluates how good the algorithms models that data set
			  metrics=['accuracy']) # check for accuracy

model.fit(x_train, y_train, epochs=5) # train model

model.evaluate(x_test, y_test, verbose=2) # evaluate the model

probability_model = tf.keras.Sequential([ # create mathematical representation with the events and their probabilities
	model, # the model itself
	tf.keras.layers.Softmax() # activation function
])

probability_model(x_test[:5]) # returns a probability of the model
