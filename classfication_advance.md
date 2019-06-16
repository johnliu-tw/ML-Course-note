### logsitic regression cost function with regularized


```
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));
h_theta = sigmoid(X * theta);

J = (1/m) * (((-y') * log(h_theta)) - ((1-y') * log(1-h_theta))) + (lambda/(2*m)) * (sum(theta(2:end) .^ 2));

temp = theta;
temp(1) = 0;

grad = ((1 / m) * X' * (h_theta - y)) + (lambda / m) * temp;
grad = grad(:);
end

```

### One-vs-all classifier

- Set data

```
X = [5000 x 400] dataset
Y = [400 x 1] dataset
num_lables = 10
lambda = 0.1
options = optimset('GradObj', 'on', 'MaxIter', 50);
```

X became [5000 x 401] dataset
initial_theta = [401 x 1]( all is 0)



- Traning data
```
for c = 1:num_labels 
    all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options)
end
```

alltheta is 10 x 401 matrix

- Predict

```
predict = sigmoid(X*all_theta')
[~, p] = max(predict, [], 2);
```
predict is 5000 x 10 matrix
max function, if parameter is 1, it would summarize each row(5000 results), if parameter is 2, it would summarize each column(10 results, the same with labels)

- Check accuracy

 mean(double(p) == y)) * 100)


### Neural Network 

- Set data 

```
X = [5000 x 400]
Theta1 = [25 x 401] (has been trained)
Theta2 = [10 x 26] (has been trained)
```

- Cauculate

```
X = [ones(m, 1) X]; # 5000 x 401
t1 = sigmoid(X * Theta1'); # 5000 x 25
t1 = [ones(m, 1) t1];# 5000 x 26

t2 = sigmoid( t1 * Theta2'); # 5000 x 10

[~, p] = max(t2, [], 2); # max value of each vector
```



