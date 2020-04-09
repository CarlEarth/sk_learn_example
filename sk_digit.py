from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve,train_test_split
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np

####switch####

model=0 #KNN = 0, SVC=1
Ex1=0   #Ex1: Split data to train the model. 
	#After training, calculate the accurracy by test data. 
	#Show the wrong-prediction results.
Ex2=0 	#Ex2: Cross_validation
	#Use cross validation method to score the model. 
	#Find the parameter value with best performance 
Ex3=0   #Ex3: Learning curve
	#See the procession of learning.
	#We can observe if there is the overfiting.
Ex4=0   #Use the model which has been trained and saved.
Ex5=1 	#Compare SVC and KNN models in learning curve.

##############
#Load data
digits = load_digits() #This data is for the number by handwriting.
X = digits.data
Y = digits.target
images = digits.images

if (Ex1==1):
	#Ex1: Split data to train the model. 
	#After training, calculate the accurracy by test data. 
	#Show the wrong-prediction results.

	#Split data to train the model. Train_data:Test_data = 8:2
	x_train, x_test, y_train, y_test, i_train, i_test = train_test_split(X, Y, images,test_size=0.2)
	if (model == 0 ):#KNN
		KNN = KNeighborsClassifier(n_neighbors=2)
		KNN.fit(x_train, y_train)
		title='KNN'
		print ("The prediction for the test data:",KNN.predict(x_test[:10]))
		print ("The target for the test data:    ",y_test[:10])
		num_success = sum(KNN.predict(x_test) == y_test)
		accuracy = num_success/len(y_test)
		num_failed=len(y_test)-num_success
		print ("accuracy=",accuracy)
		joblib.dump(KNN, 'save/KNN_digit.pkl') # Save
		fig = plt.figure(figsize=(num_failed, 1))# 調整子圖形 
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
		j=0
		for i in range(0,len(y_test)):
			if(KNN.predict([x_test[i]]) != y_test[i]):
				# 在 1 x num_failed 的網格中第 i + 1 個位置繪製子圖形，並且關掉座標軸刻度
				ax = fig.add_subplot(1, num_failed, j + 1, xticks = [], yticks = [])
				# 顯示圖形，色彩選擇灰階
				ax.imshow(i_test[i], cmap = plt.cm.binary)
				# 在左下角標示目標值
				ax.text(0, 7, str(y_test[i]))
				ax.text(0, 5, str(KNN.predict([x_test[i]])))
				j=j+1			
		plt.show()
	else: #SVC
		svc=SVC(gamma=0.001)
		svc.fit(x_train, y_train)
		title='SVC'
		print ("The prediction for the test data:",svc.predict(x_test[:10]))
		print ("The target for the test data:    ",y_test[:10])
		num_success = sum(svc.predict(x_test) == y_test)
		accuracy = num_success/len(y_test)
		num_failed=len(y_test)-num_success
		
		print ("accuracy=",accuracy)
		joblib.dump(svc, 'save/SVC_digit.pkl') # Save
		fig = plt.figure(figsize=(num_failed, 1))# 調整子圖形 
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
		j=0
		for i in range(0,len(y_test)):
			if(svc.predict([x_test[i]]) != y_test[i]):
				# 在 1 x num_failed 的網格中第 i + 1 個位置繪製子圖形，並且關掉座標軸刻度
				ax = fig.add_subplot(1, num_failed, j + 1, xticks = [], yticks = [])
				# 顯示圖形，色彩選擇灰階
				ax.imshow(i_test[i], cmap = plt.cm.binary)
				# 在左下角標示目標值
				ax.text(0, 7, str(y_test[i]))
				ax.text(0, 5, str(svc.predict([x_test[i]])))
				j=j+1			
		plt.show()

if (Ex2==1):

	#The method 2:
	#Use cross_validation KNeighborsClassifier model to find the best parameter
	if (model == 0 ):
		param_range = range(1, 10)
		train_score, test_score = validation_curve(
			KNeighborsClassifier(), X, Y, param_name='n_neighbors', param_range=param_range, cv=5,
			scoring='accuracy')
		xlabel="Number of neighbor"
		title='KNN for digits data with different n_neighbor'
	else:
		param_range = np.logspace(-6, -2.3, 15)
		train_score, test_score = validation_curve(
			SVC(), X, Y, param_name='gamma', param_range=param_range, cv=5,
			scoring='accuracy')
		xlabel="gamma"
		title='SVC model for digits data with different gamma'

	train_score_mean = np.mean(train_score, axis=1)
	test_score_mean = np.mean(test_score, axis=1)

	plt.plot(param_range, train_score_mean, 'o-', color="r",
		     label="Training")
	plt.plot(param_range, test_score_mean, 'o-', color="g",
		     label="Cross-validation")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel("score")
	plt.legend(loc="best")
	plt.show()

if (Ex3==1):

	#The method 3:
	#Use learning curve to see if there is overfit phenomena,
	if (model == 0 ):
		train_sizes, train_score, test_score= learning_curve(
		KNeighborsClassifier(n_neighbors=2), X, Y, cv=6, scoring='accuracy',
		train_sizes=[0.2, 0.4, 0.6, 0.8, 1])
	else:
		train_sizes, train_score, test_score= learning_curve(
		SVC(gamma=0.01), X, Y, cv=6, scoring='accuracy',
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

	KNN2 = joblib.load('save/KNN_digit.pkl')
	svc2 = joblib.load('save/SVC_digit.pkl')
	xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.1)
	print("The KNN prediction for the test data:",KNN2.predict(xtest[:10]))
	print("The SVC prediction for the test data:",svc2.predict(xtest[:10]))
	print("The target for the test data:        ",ytest[:10])

if (Ex5==1):

	train_sizes1, train_score1, test_score1= learning_curve(
		KNeighborsClassifier(n_neighbors=2), X, Y, cv=6, scoring='accuracy',
		train_sizes=[0.2, 0.4, 0.6, 0.8, 1])
	train_sizes2, train_score2, test_score2= learning_curve(
		SVC(gamma=0.001), X, Y, cv=6, scoring='accuracy',
		train_sizes=[0.2, 0.4, 0.6, 0.8, 1])
		
	train_score_mean1 = np.mean(train_score1, axis=1)
	test_score_mean1 = np.mean(test_score1, axis=1)
	train_score_mean2 = np.mean(train_score2, axis=1)
	test_score_mean2 = np.mean(test_score2, axis=1)


	plt.plot(train_sizes1, test_score_mean1, 'o-', color="r",
		     label="KNN")
	plt.plot(train_sizes2, test_score_mean2, 'o-', color="g",
		     label="SVC")

	plt.xlabel("Training examples")
	plt.ylabel("score")
	plt.legend(loc="best")
	plt.show()
