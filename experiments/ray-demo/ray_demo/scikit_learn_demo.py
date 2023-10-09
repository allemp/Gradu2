# %%  SGD regressor, SGD classifier
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
X, y = make_classification(
    n_samples=1000000,
    n_features=50,
    n_informative=30,
    n_classes=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
# %%
scaler = StandardScaler()

scaler.fit(X_train)
# %%
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
# %%
classifier = SGDClassifier(max_iter=1000, tol=1e-3)
# %%
batch_size = 100
metrics = []
for i in range(0, len(X_train), batch_size):
    X_batch = scaled_X_train[i : i + batch_size]
    y_batch = y_train[i : i + batch_size]
    classifier.partial_fit(X_batch, y_batch, classes=np.unique(y))
    predicions = classifier.predict(scaled_X_test)
    # loss = classifier.Lo
    accuracy = accuracy_score(predicions, y_test)
    train_loss = hinge_loss(y_batch, classifier.decision_function(X_batch))
    test_loss = hinge_loss(y_test, classifier.decision_function(scaled_X_test))
    metrics.append(
        {
            "iteration": i,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
        }
    )

# %%
df = pd.DataFrame(metrics)
# %%
df.plot(x="iteration", y=["train_loss", "test_loss"], ylim=(0, 2))

# %%
