### Anomaly Detection and Collaborative Filtering
- Data Init

```
X = [1;2;3;4]
```

- Estimate Gaussian

```
m = 4
mu = 1/m * sum(X); // mu=2.5

sigma2 = 1/m * sum((X - repmat(mu, m, 1)).^2);

// X - repmat(mu, m, 1) = [-1.5, -0.5, 0.5, 1.5]
```

- Find Outliers

step1: get step size

```
stepsize = (max(pval) - min(pval)) / 1000;
```

step2: enumerate according to epsilon

```
for epsilon = min(pval):stepsize:max(pval)
```

step3: make prediction

```
predictions = (pval < epsilon);
```

step4: set formula

##### fp 
the number of false positives: the ground truth label says it’s not an anomaly, but our algorithm incorrectly classified it as an anomaly

##### fn
the number of false negatives: the ground truth label says it’s an anomaly, but our algorithm incorrectly classified it as not being anomalous.

##### tp
the number of true positives: the ground truth label says it’s an anomaly and our algorithm correctly classified it as an anomaly.

```
fp = sum((predictions == 1) & (yval == 0));
fn = sum((predictions == 0) & (yval == 1));
tp = sum((predictions == 1) & (yval == 1));

prec = tp / (tp + fp);
rec = tp / (tp + fn);

F1 = 2 * prec * rec / (prec + rec);
```

step5: get best F1 & epsilon

```
if F1 > bestF1
   bestF1 = F1;
   bestEpsilon = epsilon;
end
```

- collaborative filtering

step1: Get J, gradiant and Theta_grad

```
J = 1/2 * sum(sum((R.* ((X*Theta') - Y)).^2));
X_grad = (R .* (X*Theta' - Y)) * Theta;
Theta_grad = (R .* (X*Theta' - Y))' * X;
```

step2: With regularization

```
J = J + lambda/2 * (sum(sum(Theta.^2)) + sum(sum(X.^2)));
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;
```