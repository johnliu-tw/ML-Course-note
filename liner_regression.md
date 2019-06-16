### Multi variable cost function

```octave
function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = (1/(2*m)) * (X * theta - y)' * (X * theta - y); % Vectorized

end
```

### Multi gradientDescent

```octave
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    delta = (1/m)*sum(X.*repmat((X*theta - y), 1, size(X,2)));
    
    
    theta = (theta' - (alpha * delta))';

    J_history(iter) = computeCostMulti(X, y, theta);

end

end
```