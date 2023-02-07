from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = [["Sepal_Length", "Sepal_Width","Petal_Length","Petal_Width"]])
print(iris_df)

target = pd.DataFrame(iris.target)
full_df = pd.concat([iris_df,target], axis=1)
full_df.columns = ['Sepal Length', "Sepal Width", "Petal Length", "Petal Width", 'Target']



# The indices of the features that we are plotting (class 0 & 1)

# this formatter will label the colorbar with the correct target names
x = full_df[['Petal Length','Petal Width']].values
y = full_df['Target'].values

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x,y)

x_new = np.array([[1.7, 0.2], [1.6,0.3], [5.1,1.8], [3.3,0.8]])

predictions = knn.predict(x_new)
print(predictions)



# this formatter will label the colorbar with the correct target names
x_index = 2
y_index = 3
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,4))
plt.scatter(x_new[:,0], x_new[:,1], c=predictions)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.tight_layout()
plt.show()
