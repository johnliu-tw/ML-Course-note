


### 1. Data Structure 

X = [
    [1,3]
    [4,5]
    ..
]

Y = [4,
     5
]

### 2. Compute Cost and Gradient

- Add intercept term to x (add one on front of every element)

X = [
    [1,1,3]
    [1,4,5]
    ..
]

- Create initial theta(size is original features + 1)

initial theta = [
    [0],
    [0],
    [0]
]

- Call cost Function

[J, grad] = costFunction(initial_theta, X, y);

```objectivec
function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
grad = zeros(size(theta));
h_theta = sigmoid(X*theta);
J = (1 / m) * ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta));
grad = (1 / m) * (h_theta - y)' * X;
end

function g = sigmoid(z)
g = 1 ./ (1 + exp(-z));
end
```

- Use Test theta

test_theta = [-24; 0.2; 0.2]

[cost, grad] = costFunction(test_theta, X, y);

- Optimizing using fminunc ( Octave function )

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

- Plot Boundary 

plotDecisionBoundary(theta, X, y);

- Use new theta to predict and get probability 

prob = sigmoid([1 45 85] * theta);

p = predict(theta, X);

```objectivec
function p = predict(theta, X)
m = size(X, 1); % Number of training examples
p = zeros(m, 1);
p = sigmoid(X * theta) >= 0.5;
end
```

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('Expected accuracy (approx): 89.0\n');

fprintf('\n');


