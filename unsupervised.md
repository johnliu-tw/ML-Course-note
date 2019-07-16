### Unsupervised learning

- Set data

```
X = [15 x 11]
centroids = [5 x 11]
```

- Step1: Find the distance from each row of X to each row of centroid

eg: 
X = [1,2,3;4,5,6]
centroids = [1,2,3]

count 1-1, 2-2, 3-3, then 4-1, 4-5, 4-6.....

```
K = size(centroids, 1);
m = size(X, 1)

for i = 1:m
    # make empty distance array
    distance_array = zeros(1,K);
    for j = 1:K
    	  # store distance value to distance_array
        distance_array(1,j) = sqrt(sum(power((X(i,:)-centroids(j,:)),2)));
    end
    
    # find a minimum value from which centroid
    [~, d_idx] = min(distance_array);
    idx(i,1) = d_idx;
end

idx #display each row of X correspond which centroid

```

- step2: Get the mean of each centroid

Extract the each cluster of X, and mean all the value to get centrol centroid.
eg: k=2, eatract x(1), x(3) and x(11), then get centrol value 

```
for k = 1:K 
   centroids(k, :) = mean(X(idx==k, :));
end

```

### PCA

- Set data

```
X = [4 x 2]
```

- step1: Get U, S, V

```
# should normalize X
Sigma = 1/m .* X' * X;
[U, S, V] = svd(Sigma);
```
U = [4x4] 
S = [4x2]
V = [2x2]

- step2: Project Data

if k = 1(dimension = 1)

```
for i=1:size(X, 1),
    for j=1:K,
        x = X(i, :)';
        projection_k = x' * U(:, j);
        new_X(i, j) = projection_k;
    end
end
```

- step3: Recover Data

```
X_rec = zeros(size(new_X, 1), size(U, 1));
for i = 1:size(Z,1),
    for j = 1:size(U,1),
        v = Z(i, :)'
        recovered_j = v' * U(j, 1:K)'
        X_rec(i, j) = recovered_j
    end
end
```