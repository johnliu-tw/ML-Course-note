### Regularized Linear Regression Cost
- Step1: Data init

```
X = [12*1]
y = [12*1]
theta = [2*1]
m = length(y);
lambda = 0;

```

- Step2: Get J cost value

```
# linearRegCostFunction

X = [ones(m, 1) 12*1] # [12*2]
h_theta = X * theta;

J = 1/(2*m) * (h_theta - y)' * (h_theta - y) + ...
    (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));
```

- Step3: Gradient

```
thetaZero = theta;
thetaZero(1) = 0;

grad = ((1 / m) * (h_theta - y)' * X) + ...
    lambda / m * thetaZero';

grad = grad(:);

```

### Error Analysis
##### Get learning curve data

1. Split train data from 1 to the entire training dataset size.
2. This curve will display the relation for numbers of training examples. 

Xval and Yval are validation dataset

```
for i = 1:m
    X_sub = X(1:i, :);
    y_sub = y(1:i); 

    theta = trainLinearReg(X_sub, y_sub, lambda);
    error_train(i) = linearRegCostFunction(X_sub, y_sub, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
    
end
```


```
function [theta] = trainLinearReg(X, y, lambda)

initial_theta = zeros(size(X, 2), 1); 
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
options = optimset('MaxIter', 200, 'GradObj', 'on');
theta = fmincg(costFunction, initial_theta, options);

end
```

##### Get PolyFeatures

```
p = 8
X_poly = zeros(numel(X), p); #[12x8]
m = size(X, 1);

for i = 1:m
    poly_feature = zeros(p, 1);
    for j = 1:p
        poly_feature(j) = X(i).^j
    end
    X_poly(i, :) = poly_feature;
end
```

##### Try Best lambda

```
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
for j = 1:length(lambda_vec)
    theta = trainLinearReg(X, y, lambda_vec(j));
    error_train(j) = linearRegCostFunction(X, y, theta, 0);
    error_val(j) = linearRegCostFunction(Xval, yval, theta, 0);      
end

```