from sklearn.datasets import fetch_openml

# mnist.data (70000 images, 784 dimensions/features of 28x28 pixels)
# mnist.target (labels = integers 0-9)
mnist = fetch_openml('mnist_784')

# split data into training set (6/7) and testing set (1/7)
from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# import and apply PCA
from sklearn.decomposition import PCA

# Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)

# find how many components PCA has after fitting the model
print(pca.n_components_) # 95% of variance amounts to 330 principle components

# apply mapping (transform) the training and testing sets
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# apply logistic regression to the transformed data
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)

# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

# Predict for Multiple Observations (images)
logisticRegr.predict(test_img[0:10])

# Measure model performance
logisticRegr.score(test_img, test_lbl)
# conclusion: PCA speeds up the fitting of ML algorithms

