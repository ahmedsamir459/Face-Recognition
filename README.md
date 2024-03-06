# **Face Recognition Using PCA & LDA Algorithms for Dimensionality Reduction (Eigen Faces)**

## PCA
Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to extract important features from high-dimensional datasets. PCA works by identifying the principal components of the data, which are linear combinations of the original features that capture the most variation in the data

## LDA
Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset while maintaining the class separability. LDA is a supervised technique, meaning that it uses the class labels to perform the dimensionality reduction. LDA is a popular technique for dimensionality reduction in the field of pattern recognition and machine learning

## Dataset
- The dataset for this project is the [AT&T Face Database](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)
- The dataset is open-source and can be downloaded from Kaggle.
- The dataset contains 400 images of 40 people, each person has 10 images.
- The images are of size 92x112 pixels & in grayscale.
- The images are stored in the `datasets/faces` folder.


### Generating the Data Matrix and the Label vector

---

- The Data Matrix is 400 x 10304 where each image is flattened in a vector and saved as a row in the Matrix
- Each 10 images of a person are given the same label from [1:40]

```python
paths = ["datasets/s" + str(i) for i in range(1, 41)]
cnt = 0
Data = np.zeros((400, 10304))
labels = np.zeros((400, 1))
for i in range(40):
    labels[i * 10 : (i + 1) * 10] = i + 1
for path in paths:
    files = os.listdir(path)
    for file in files:
        img = Image.open(path + "/" + file)
        np_img = np.array(img)
        np_img = np_img.flatten()
        Data[cnt] = np_img
        cnt += 1
```

### Spliting the Dataset into Training and Test sets

---

- Keeping the odd rows (assuming row[0] is the odd) for Training and the even rows (assuming row[1] is even) for Testing

```python
X_train = Data[0::2]
X_test = Data[1::2]
y_train = labels[0::2]
y_test = labels[1::2]
```

## PCA
### pseudocode for PCA

```python
def get_PCA(X_train, alpha):
    # Compute the mean of the training data
    mean_face = np.mean(X_train, axis=0)
    # subtract the mean from the training data
    X_train_centralized = X_train - mean_face
    # compute the covariance matrix
    cov_matrix = X_train_centralized @ X_train_centralized.T
    # compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # sort the eigenvectors descindigly by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # restore the original eigenvectors
    eigenvectors_converted = X_train_centralized.T @ eigenvectors
    # normalize the eigenvectors_converted
    eigenfaces = eigenvectors_converted / np.linalg.norm(eigenvectors_converted, axis=0)
    # compute the number of components to keep
    sum = 0
    no_components = 0
    for i in range(len(eigenvalues)):
        sum += eigenvalues[i]
        no_components += 1
        if sum / np.sum(eigenvalues) >= alpha:
            break
    # project the training data on the eigenfaces
    return  mean_face, eigenfaces[:, :no_components]
```

**Trick to get eigen values/vectors from Cov Matrix** <br>

1. The Cov Matrix is Computed as Z.T \* Z (10304 x 10304) so getting the eigen values/vectors from this matrix requires too much time.
2. Instead we computed Cov matrix as Z \* Z.T, According to Linear Algebra the eigen values computed from this matrix is the same as the original one but takes only the first 200 (which covers over 99% of the total variance).
3. Where the original eigen vectors are restored by this formula: ui=A\*vi where ui is the original eigen vector (10304 x 1) and vi is the smaller one (200 x 1).
4. It gives the same results and in lesser time.

#### The First 5 eigen faces

---

![alt text](images/image.png)

#### Projecting The Train Data and Test Data using the Same Projection Matrix

---

```python
def PCA_Projected_data(mean_face,eigenfaces):
    X_train_centered = X_train - mean_face
    X_train_projected = X_train_centered @ eigenfaces
    X_test_centered = X_test - mean_face
    X_test_projected = X_test_centered @ eigenfaces
    return X_train_projected, X_test_projected
```

#### Using KNN with K=1 as a classifier

---

- The Classifier is trained with the projected training data using **knn.fit()**
- Then the classifier is given the projected test data and the predicted values (labels) are saved in **Y_pred**
- The y_pred is compared with the y_test to get accuracy (actual labels)

```python
def Test_PCA(alpha, k):
    mean_face, eigenfaces = get_PCA(X_train, alpha)
    X_train_pca, X_test_pca = PCA_Projected_data(mean_face, eigenfaces)
    knn = KNeighborsClassifier(k, weights="distance")
    knn.fit(X_train_pca, y_train.ravel())
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy
```

#### The Accuracy for each value of alpha

---

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>0.945</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.94</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.94</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.93</td>
    </tr>
  </tbody>
</table>
</div>

- As alpha increases, the classification accuracy decreases due to overfitting and it is more sensitive to noises
  ![alt text](images/image-1.png)

## LDA
- Introducing our second trick 
- Regularization step: before computing the inverse of the S_W matrix we add a value of 1e-7 to the diagonal of the matrix.
-It is a common practice when dealing with the inversion of matrices in the context of Linear Discriminant Analysis (LDA) or covariance matrices in general. It is often used to ensure numerical stability, especially when the matrices involved might be close to singular.
`S_W += 1e-7 * np.identity(X_train.shape[1])`
### pseudocode for LDA

```python
def get_LDA(X_train, y_train):
    y_train = np.squeeze(y_train)
    class_means = np.array([np.mean(X_train[y_train == i], axis=0) for i in range(1, 41)])
    class_sizes = np.array([np.sum(y_train == i) for i in range(1, 41)])

    # Compute overall mean
    overall_mean = np.mean(X_train, axis=0)

    # Compute within-class scatter matrix
    S_W = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(1, 41):
        # Use boolean index to select rows from X_train
        class_data = X_train[y_train == i]
        centered_data = class_data - class_means[i - 1]
        S_W += np.dot(centered_data.T, centered_data) 

    # # Regularize S_W
    S_W += 1e-7 * np.identity(X_train.shape[1])

    # Compute between-class scatter matrix
    S_B = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(1, 41):
        # Use boolean index to select rows from X_train
        class_data = X_train[y_train == i]
        class_diff = class_means[i - 1] - overall_mean
        S_B += class_sizes[i - 1] * np.outer(class_diff, class_diff)

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Take only the dominant eigenvectors
    projection_matrix = sorted_eigenvectors[:, :39]
    return np.real(projection_matrix)
```

#### Projecting The Train Data and Test Data using the Same Projection Matrix

---

```python
def LDA_projected_data(projection_matrix):
    projected_X_train = np.dot(X_train, projection_matrix)
    projected_X_test = np.dot(X_test, projection_matrix)
    return projected_X_train, projected_X_test
```

#### Using KNN with K=1 as a classifier

---

- The Classifier is trained with the projected training data using **knn.fit()**
- Then the classifier is given the projected test data and the predicted values (labels) are saved in **Y_pred**
- The y_pred is compared with the y_test to get accuracy (actual labels)

```python
def Test_LDA(k):
    projected_X_train, projected_X_test = LDA_projected_data(LDA_projection_matrix)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(projected_X_train, y_train.ravel())
    y_pred = knn.predict(projected_X_test)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy
```

- LDA Accuracy for k = 1 is **95%**

## Classifier Tunning (Hyperparameter Tuning for K in KNN)

### The tie breaking is done by choosing the least distance (weights = "distance")

- PCA
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>7</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>0.945</td>
      <td>0.9</td>
      <td>0.895</td>
      <td>0.88</td>
      <td>0.835</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.94</td>
      <td>0.9</td>
      <td>0.895</td>
      <td>0.86</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.94</td>
      <td>0.905</td>
      <td>0.895</td>
      <td>0.85</td>
      <td>0.815</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.93</td>
      <td>0.9</td>
      <td>0.865</td>
      <td>0.83</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>

- LDA
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.915</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.890</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.875</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.860</td>
    </tr>
  </tbody>
</table>
</div>

### Variations
---

#### PCA Variant (Kernel PCA)

- Kernel PCA is a non-linear dimensionality reduction technique that uses a kernel function to map high-dimensional data into a lower-dimensional space. This allows it to capture non-linear relationships between variables that are not possible with linear PCA.
- The time complexity of normal PCA is O(d^3), where d is the number of dimensions, while the time complexity of kernel PCA is O(n^3), where n is the number of data points. The computation of the kernel matrix is the most computationally expensive step in kernel PCA.
- Kernel PCA may be more accurate than normal PCA for datasets with non-linear relationships between variables, as it can capture these relationships. However, kernel PCA is more prone to overfitting than normal PCA, and the choice of kernel function can greatly affect the performance of kernel PCA.
```python
def Test_Polynomial_Kernel_PCA(no_of_components, k):
    kpca = KernelPCA(n_components=no_of_components, kernel="poly")
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    knn = KNeighborsClassifier(k, weights="distance")
    knn.fit(X_train_kpca, y_train.ravel())
    y_pred = knn.predict(X_test_kpca)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy
```
- Comparison Between Normal PCA and Kernel PCA

![alt text](images/PCA_vs_kerneL_PCA.png)


#### LDA Variant (Shrinkage LDA)

- Shrinkage LDA (Linear Discriminant Analysis) is a variant of the standard LDA method that is used for classification and dimensionality reduction. The key difference between shrinkage LDA and normal LDA is that the former incorporates a regularization term that shrinks the sample covariance matrix towards a diagonal matrix.

- This regularization is particularly useful when dealing with high-dimensional data, as it helps to overcome the small sample size problem by stabilizing the covariance estimates. Shrinkage LDA has been shown to outperform traditional LDA in terms of classification accuracy, especially when the number of features is much larger than the number of observations.

- Another advantage of shrinkage LDA is that it can handle multicollinearity between the predictor variables, which can be a problem in standard LDA when the predictors are highly correlated. In summary, shrinkage LDA is a powerful tool for classification and dimensionality reduction that can improve the accuracy of LDA in high-dimensional and small sample size settings.
```python
def Test_Shrinkage_LDA(k):
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X_train, y_train.ravel())
    X_train_projection = lda.transform(X_train)
    X_test_projection = lda.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_projection, y_train.ravel())
    y_pred = knn.predict(X_test_projection)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy
```
- Comparison Between Normal LDA & Shrinkage LDA

![alt text](images/lda_vs_shrinkage_lda.png) 

### Splitting The Dataset into 70% Training and 30% Testing
---
- The split function
```python
def split_data(data,labels):
    bonus_x_train = np.zeros((280,10304))
    bonus_x_test = np.zeros((120,10304))
    bonus_y_train = np.zeros((280,1))
    bonus_y_test = np.zeros((120,1))
    # split each person's data into 7 training images and 3 testing images
    for  i in range (40):
        bonus_x_train[i*7:(i+1)*7] = data[i*10:i*10+7]
        bonus_x_test[i*3:(i+1)*3] = data[i*10+7:i*10+10]
        bonus_y_train[i*7:(i+1)*7] = labels[i*10:i*10+7]
        bonus_y_test[i*3:(i+1)*3] = labels[i*10+7:i*10+10]
    # shuffle the data
    indices = np.arange(280)
    np.random.shuffle(indices)
    bonus_x_train = bonus_x_train[indices]
    bonus_y_train = bonus_y_train[indices]   
    return bonus_x_train,bonus_x_test,bonus_y_train,bonus_y_test
```
#### Comparison Between Ordinary Split and Train Split in PCA
---
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Original Split</th>
      <th>Bonus Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>0.945</td>
      <td>0.966667</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.94</td>
      <td>0.958333</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>0.94</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.93</td>
      <td>0.941667</td>
    </tr>
  </tbody>
</table>
</div>    

![alt text](images/Bonus_PCA.png)
- It is clear that after splitting the dataset to 70/30 the accuracy is better for all values of **alpha**

#### Comparison Between Ordinary Split and Train Split in LDA
---
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Original Split</th>
      <th>Bonus Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LDA</th>
      <td>0.95</td>
      <td>0.975</td>
    </tr>
  </tbody>
</table>
</div>

![alt text](images/Bonus_LDA.png)


# Face vs. Non-Face Classification using PCA and LDA

## Introduction

This documentation outlines the implementation of Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for the task of classifying face and non-face images. The goal is to explore the effectiveness of dimensionality reduction techniques and assess the model's performance. Comparing the impact of non-faces number in the accuracy of the model.


## Implementation Details

### Libraries Used
- NumPy
- Scikit-learn
- Images

### Data Preprocessing

#### Non-Face Images
1. Download a semi large dataset of non faces images (~550) sample.
2. Loaded non-face images from the specified file paths.
3. Convert images into gray scale
4. Resized each image to 92x112 pixels.
5. Flattened each image to a 1D array.

#### Face Images
1. Loaded face images from the specified file paths.
2. Resized each image to 92x112 pixels.
3. Flattened each image to a 1D array.

#### Labels
1. Created binary labels (1 for faces, 0 for non-faces).

#### Shuffle images within each class
1. shuffle face images and thier labels
2. shuffle non_faces images and thier labels


### Data Splitting
1. Split the data into training and testing sets using `split_data`.
2. Combined face and non-face data.
3. Re-shuffle the combined dataset 

-code 

```python
def split_data(faces, faces_labels, non_faces,non_faces_labels,non_faces_count,alpha,non_face_precentage_in_train=1):
    if alpha == 0.5:
        faces_train = faces[::2]
        faces_train_labels = faces_labels[::2]
        faces_test = faces[1::2]
        faces_test_labels = faces_labels[1::2]
        non_faces_train = non_faces[:int(non_faces_count*non_face_precentage_in_train):2]
        non_faces_train_labels = non_faces_labels[:int(non_faces_count*non_face_precentage_in_train):2]
        non_faces_test = non_faces[1:non_faces_count:2]
        non_faces_test_labels = non_faces_labels[1:non_faces_count:2]
    else:
        n = len(faces)
        n_train = int(n*alpha)
        idx = np.random.permutation(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        faces_train = faces[train_idx]
        faces_train_labels = faces_labels[train_idx]
        faces_test = faces[test_idx]
        faces_test_labels = faces_labels[test_idx]
        n = non_faces_count
        n_train = int(n*alpha)
        idx = np.random.permutation(n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        non_faces_train = non_faces[train_idx]
        non_faces_train_labels = non_faces_labels[train_idx]
        non_faces_test = non_faces[test_idx]
        non_faces_test_labels = non_faces_labels[test_idx]
    
    return np.append(faces_train, non_faces_train, axis=0), np.append(faces_train_labels, non_faces_train_labels, axis=0), np.append(faces_test, non_faces_test, axis=0), np.append(faces_test_labels, non_faces_test_labels, axis=0)

```

- sample of training set
![alt text](images/trainplot.png)
- sample of test set
![alt text](images/image-2.png)


### Dimensionality Reduction: PCA

1. Applied PCA for dimensionality reduction.
2. Explored the variance explained by different components.
3. Transformed both training and testing data using the selected number of components.
4. Use the most efficient alpha value from the first part `0.85`.

- code 
    ```python
    def PCA(train_data,alpha=0.85):
        mean = np.mean(train_data, axis=0)
        centered_data = train_data - mean
        cov_matrix = np.dot(centered_data,centered_data.T)
        eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        eig_vectors = np.dot(centered_data.T,eig_vectors)
        for i in range(eig_vectors.shape[1]):
            eig_vectors[:,i] = eig_vectors[:,i]/np.linalg.norm(eig_vectors[:,i])
        total = np.sum(eig_values)
        k = 0
        var = 0
        while var/total < alpha:
            var += eig_values[k]
            k += 1
        return eig_vectors[:,:k], mean

    def project_data(data, eigenvectors, mean,):
        return np.dot(data - mean, eigenvectors)
    ```
### Dimensionality Reduction: LDA

1. Applied LDA for dimensionality reduction.
2. Use only one dominant eigenvector as we have only two classes.
3. Transformed both training and testing data using the selected number of components.

- code 
    ```python
    def LDA (train_data, train_labels, k=1):]
        mean1 = np.mean(train_data[train_labels.ravel() == 1], axis=0)
        mean0 = np.mean(train_data[train_labels.ravel() == 0], axis=0)

        Sw = np.dot((train_data[train_labels.ravel() == 1] - mean1).T, 
                    (train_data[train_labels.ravel() == 1] - mean1)) 
                + np.dot((train_data[train_labels.ravel() == 0] - mean0).T, 
                            (train_data[train_labels.ravel() == 0] - mean0))
        Sb = np.dot((mean1 - mean0).reshape(-1,1), (mean1 - mean0).reshape(-1,1).T)

        eig_values, eig_vectors = np.linalg.eigh(np.dot(np.linalg.inv(Sw), Sb))
        eig_values = np.real( eig_values)
        eig_vectors = np.real( eig_vectors)
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        return eig_vectors[:,:k]
    ```

### Model Training and Evaluation

1. Use KNN for training and evaluation, and stick to `k=1` for training and `weight='distance'`, so it weight points by the inverse of their distance.
2. Trained the model using the transformed data.
3. Evaluated the model using accuracy.

- code 
    ```python
    def knn_classifier(train_data, train_labels, test_data, test_labels, k=1):
        knn = KNeighborsClassifier( n_neighbors=1, weights='distance')
        knn.fit( train_data, train_labels.ravel() )
        return accuracy_score(test_labels, knn.predict(test_data).ravel()),knn.predict(test_data).ravel()

    ```
### Results

#### PCA Results

- On average it takes **`41` components** to retain 85.0% of the variance.
- On average the **accuracy** of the model is about **`94.5%`**.
- The maximum accuracy of the model is

    ![alt text](images/image-7.png)

#### LDA Results

- Only use one of dominant eigenvectors.
- On average the **accuracy** of the model is about **`80%`**.

### Visualizing Results and Comparisons

#### Showed failure and success cases with visualizations.
- **For PCA**

    ![alt text](images/image-4.png)
- **For LDA**

    ![alt text](images/image-5.png)


#### Plotted accuracy vs the number of non-face images while fixing the number of face images.
- **For PCA**

    ![alt text](images/image-6.png)
- **For LDA**

    ![alt text](images/ldacurve.png)

    While the number of non faces images increases the accuracy of the model decreases as it biased to recognize face images as non-face images. Also due to the increase of the data, the number of points increases and the gap between classes decreases so it make some noise and confusion for the classifier.

#### Criticize the accuracy measure for large numbers of non-faces images in the training data.
- **For PCA** 

    ![alt text](images/image-3.png)
- **For LDA**

    ![alt text](images/image-8.png)

    while the number of non-faces images in the training set increases the accuracy of the model increases as the number of non-faces images in the the test set fixed. So the model train on a lower number of non-faces images accoring to the test ones. 

## Contributers
- Mohamed Wael Mohamed
<br>
- Ahmed Samir Said
<br>
- Youssef Hossam Aboelwafa
