import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold

from matplotlib import pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, y_train, X_test, y_test, n_splits=5):
    # Initialize cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=43)
    
    
    # Binarize the output labels for multiclass ROC AUC
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
    n_classes = y_train_bin.shape[1]
    
    
    # Compute cross-validation ROC AUC score
    cv_scores = cross_val_score(model, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), 
                                cv=cv_strategy, scoring='roc_auc_ovr', n_jobs=-1)
    mean_cv_score = cv_scores.mean()
    print(f'Mean CV ROC AUC Score: {mean_cv_score}')
    
    
    # for ROC AUC
    y_proba = model.predict_proba(X_test)

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    print(f'Test Data AUC ROC: {roc_auc}')
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # for classification report and matrix
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()