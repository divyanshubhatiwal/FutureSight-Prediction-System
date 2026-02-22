import pickle
import os
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import streamlit as st

def checkDir():
	if 'models' in os.listdir('../'): 
		return True
	return False

def makeDir():
	if checkDir(): 
		pass
	else: 
		os.mkdir('../models')

# will save a model at ../models and will return the location+name of saved model
def saveModel(modelClass, name = None):
	fileName = name
	if name is None: 
		fileName = 'model'+str(len(os.listdir('../models')))
	fileName+='.sav'
	pickle.dump(modelClass, open('../models/'+fileName, 'wb'))
	return '../models/'+fileName

# model will be loaded through the location of model that is returned from the 
def loadModel(fileName):
	model = pickle.load(open(fileName, 'rb'))
	return model

def plot_regression_results(y_test, y_pred, model_name):
	"""Plot actual vs predicted values for regression"""
	fig, ax = plt.subplots(figsize=(10, 6))
	
	# Scatter plot
	ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
	
	# Perfect prediction line
	min_val = min(y_test.min(), y_pred.min())
	max_val = max(y_test.max(), y_pred.max())
	ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
	
	ax.set_xlabel('Actual Values', fontsize=12)
	ax.set_ylabel('Predicted Values', fontsize=12)
	ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	return fig

def plot_confusion_matrix(y_test, y_pred, model_name, class_names=None):
	"""Plot confusion matrix for classification"""
	cm = confusion_matrix(y_test, y_pred)
	
	fig, ax = plt.subplots(figsize=(8, 6))
	
	# Create heatmap
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
	            xticklabels=class_names if class_names else 'auto',
	            yticklabels=class_names if class_names else 'auto')
	
	ax.set_xlabel('Predicted Label', fontsize=12)
	ax.set_ylabel('True Label', fontsize=12)
	ax.set_title(f'{model_name}: Confusion Matrix', fontsize=14, fontweight='bold')
	
	return fig

def plot_feature_importance(model, feature_names, model_name, top_n=10):
	"""Plot feature importance for models that support it"""
	try:
		# Get feature importance or coefficients
		if hasattr(model, 'feature_importances_'):
			importances = model.feature_importances_
			title_suffix = 'Feature Importance'
		elif hasattr(model, 'coef_'):
			importances = np.abs(model.coef_)
			if len(importances.shape) > 1:
				importances = importances[0]  # For multi-class, take first class
			title_suffix = 'Feature Coefficients (Absolute)'
		else:
			return None
		
		# Create dataframe of features and importances
		feature_imp = pd.DataFrame({
			'feature': feature_names,
			'importance': importances
		}).sort_values('importance', ascending=False).head(top_n)
		
		# Plot
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.barh(range(len(feature_imp)), feature_imp['importance'])
		ax.set_yticks(range(len(feature_imp)))
		ax.set_yticklabels(feature_imp['feature'])
		ax.invert_yaxis()
		ax.set_xlabel('Importance', fontsize=12)
		ax.set_title(f'{model_name}: {title_suffix} (Top {top_n})', fontsize=14, fontweight='bold')
		ax.grid(True, alpha=0.3, axis='x')
		
		return fig
	except Exception as e:
		st.warning(f"Could not generate feature importance plot: {e}")
		return None

### All the below tests passed
if __name__ == '__main__':
	print(checkDir())
	makeDir()
	reg = linear_model.Ridge(alpha = 0.5)
	reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
	print("og Coeff: ",reg.coef_)
	path = saveModel(reg)
	print("Model Name: "+path)
	model = loadModel(path)
	print("Loaded Model:", model.coef_)
