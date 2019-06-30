### Cost function
- Step1: Data init

```
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10;  
X = [5000*400]
y = [400*1]
lambda = 1

# flate theta
nn_params = [Theta1(:) ; Theta2(:)];
```

- Step2: Get J without regularized

```
# size(nn_params) = 10285
# Theta1 = reshape(nn_params(1:10025),25,401)
# Theta2 = reshape(nn_params(10026:end),10,26)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));



Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
X = [ones(m,1) X]; # 5000*401
z2 = Theta1 * X'; # 25*5000
a2 = sigmoid(z2); 

a2 = [ones(m,1) a2'];# 26*5000
z3 = Theta2 * a2'; # 10*5000
h_theta = sigmoid(z3); 

y_new = zeros(num_labels, m); % 10*5000
for i=1:m,
  y_new(y(i),i)=1;
end

J = (1/m) * sum ( sum ( (-y_new) .* log(h_theta) - (1-y_new) .* log(1-h_theta) ));
```

- Step3:  Get J with regularized

```
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

J = J + Reg;
```

- Step4: Back propagation

```
for t=1:m
    
    # Set a1, a2 and a3
	a1 = X(t,:); # 1*401
    a1 = a1';
	z2 = Theta1 * a1; 
	a2 = sigmoid(z2); # 25*1
    
    a2 = [1 ; a2]; # 26*1
	z3 = Theta2 * a2;
	a3 = sigmoid(z3); # 10*1
    
    # Set delta_3, delta_2
    
	delta_3 = a3 - y_new(:,t); # 10*1
	
    z2=[1; z2]; # 26*1
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); # 26*1
	delta_2 = delta_2(2:end); # skipping sigma2(0), 25*1

	# Set grad
	Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1';
    
end;

Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)
```

funtion lists: 

```
function g = sigmoidGradient(z)

g = sigmoid(z).*(1-sigmoid(z));

end
```
```
function g = sigmoid(z)

g = 1.0 ./ (1.0 + exp(-z));

end
```

- Step5: Regularization

```
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); 

grad = [Theta1_grad(:) ; Theta2_grad(:)];
```