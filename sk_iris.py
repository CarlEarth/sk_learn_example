from sklearn.datasets import load_iris
from sklearn.model_selection import validation_curve,train_test_split
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np

####switch####
Ex1=0 #
Ex2=0 #Cross_validation
Ex3=0 #learning curve
Ex4=1 #Use the model which has been trained
##############

iris = load_iris()
X = iris.data
Y = iris.target

if (Ex1==1):
	#The method 1:
	#Use KNeighborsClassifier model to train.
	#After training, save the file

	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
	KNN = KNeighborsClassifier(n_neighbors=5)
	KNN.fit(x_train, y_train)

	print ("The prediction for the test data:",KNN.predict(x_test))
	print ("The target for the test data:    ",y_test)
	accuracy = sum(KNN.predict(x_test) == y_test)/len(y_test)
	print ("accuracy=",accuracy)
	# Save
	joblib.dump(KNN, 'save/KNN.pkl')

if (Ex2==1):
	#The method 2:
	#Use cross_validation KNeighborsClassifier model to find the best parameter
	n_neighbor = range(1, 31, 3)
	train_score, test_score = validation_curve(
		KNeighborsClassifier(), X, Y, param_name='n_neighbors', param_range=n_neighbor, cv=12,
		scoring='accuracy')
	train_score_mean = np.mean(train_score, axis=1)
	test_score_mean = np.mean(test_score, axis=1)

	#plt.plot(n_neighbor, train_score_mean, 'o-', color="r",
	#	     label="Training")
	plt.plot(n_neighbor, test_score_mean, 'o-', color="g",
		     label="Cross-validation")
	plt.title('KNN for iris data with different n_neighbor')
	plt.xlabel("number of neighbor")
	plt.ylabel("score")
	plt.legend(loc="best")
	plt.show()

if (Ex3==1):
	#The method 3:
	#Use learning curve to see if there is overfit phenomena,

	train_sizes, train_score, test_score= learning_curve(
		KNeighborsClassifier(n_neighbors=5), X, Y, cv=6, scoring='accuracy',
		train_sizes=[0.2, 0.4, 0.6, 0.8, 1])
	train_score_mean = np.mean(train_score, axis=1)
	test_score_mean = np.mean(test_score, axis=1)
	plt.plot(train_sizes, train_score_mean, 'o-', color="r",
		     label="Training")
	plt.plot(train_sizes, test_score_mean, 'o-', color="g",
		     label="Cross-validation")

	plt.xlabel("Training examples")
	plt.ylabel("score")
	plt.legend(loc="best")
	plt.show()

if (Ex4==1):

	KNN2 = joblib.load('save/KNN.pkl')
	xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.1)
	print("The prediction for the test data:",KNN2.predict(xtest))
	print("The target for the test data:    ",ytest)
