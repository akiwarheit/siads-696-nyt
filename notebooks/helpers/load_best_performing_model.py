import pickle

def load_best_performing_model():
    with open('out/best_performing_cv_roc_auc_model.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

