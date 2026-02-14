import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('game_data.csv')
print("data shape:", data.shape)
print("duplicates:", data.duplicated().sum())

data = data.drop_duplicates()

X = data[['reaction_time_avg', 'accuracy', 'score', 'misses']]
y = data['skill_level']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

pipeline = Pipeline([
	('scaler', StandardScaler()),
	('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42))
])

pipeline.fit(X_train, y_train)

test_acc = pipeline.score(X_test, y_test)
print("Test Accuracy:", test_acc)

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("5-fold CV accuracy: mean=", cv_scores.mean(), "std=", cv_scores.std())

try:
	clf = pipeline.named_steps['clf']
	classes = clf.classes_
	coefs = clf.coef_
	print("Classes:", classes)
	print("Coefficients per class:")
	for cls, coef in zip(classes, coefs):
		print(cls, dict(zip(X.columns, [float(c) for c in coef])))
except Exception:
	pass

joblib.dump(pipeline, 'skill_model.pkl')
print("Model saved as skill_model.pkl")
