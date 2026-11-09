import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy import stats

sys.path.insert(0, '.')
from src.data_preprocessing import build_preprocessing_pipeline

train_df = pd.read_csv('data/processed/train.csv')
X_train = train_df.drop(columns=['Revenue'])
y_train = train_df['Revenue']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for hp_name, rf_kw in [
    ('HP Set 1 (n_estimators=200, max_depth=None, min_samples_leaf=5)', {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 5, 'random_state': 42}),
    ('HP Set 2 (n_estimators=200, max_depth=15, min_samples_leaf=2)', {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 2, 'random_state': 42})
]:
    print('======================================================================')
    print(f'=== {hp_name} ===')
    print('======================================================================')
    strategies = ['none', 'balanced', 'smote']
    res = {s: {'roc': [], 'pr': [], 'brier': [], 'prec': [], 'rec': []} for s in strategies}
    
    for strat in strategies:
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx].copy(), y_train.iloc[tr_idx].copy()
            X_val, y_val = X_train.iloc[val_idx].copy(), y_train.iloc[val_idx].copy()
            
            prep_pipe = build_preprocessing_pipeline('random_forest', include_pagevalues=True, page_count_config='original', duration_config='original')
            fe_step = prep_pipe.named_steps['feature_engineer']
            proc_step = prep_pipe.named_steps['preprocessor']
            
            kw = rf_kw.copy()
            if strat == 'balanced':
                kw['class_weight'] = 'balanced'
            
            est = RandomForestClassifier(**kw)
            steps = [('feature_engineer', fe_step), ('preprocessor', proc_step)]
            if strat == 'smote':
                steps.append(('smote', SMOTE(random_state=42)))
            steps.append(('classifier', est))
            
            pipe = ImbPipeline(steps)
            pipe.fit(X_tr, y_tr)
            
            y_prob = pipe.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            res[strat]['roc'].append(roc_auc_score(y_val, y_prob))
            res[strat]['pr'].append(average_precision_score(y_val, y_prob))
            res[strat]['brier'].append(brier_score_loss(y_val, y_prob))
            res[strat]['prec'].append(precision_score(y_val, y_pred, zero_division=0))
            res[strat]['rec'].append(recall_score(y_val, y_pred, zero_division=0))
            
    for strat in strategies:
        print(f'\nStrategy: {strat.upper()}')
        r_list = res[strat]['roc']
        b_list = res[strat]['brier']
        pr_list = res[strat]['pr']
        print(f'Fold ROC-AUCs: {[round(x, 6) for x in r_list]}')
        print(f'Mean ROC-AUC:  {np.mean(r_list):.6f} +/- {np.std(r_list):.6f}')
        print(f'Fold Briers:   {[round(x, 6) for x in b_list]}')
        print(f'Mean Brier:    {np.mean(b_list):.6f} +/- {np.std(b_list):.6f}')
        print(f'Mean PR-AUC:   {np.mean(pr_list):.6f} +/- {np.std(pr_list):.6f}')
        print(f'Mean Prec:     {np.mean(res[strat]["prec"]):.6f}')
        print(f'Mean Rec:      {np.mean(res[strat]["rec"]):.6f}')
        
    roc_bal = np.array(res['balanced']['roc'])
    roc_none = np.array(res['none']['roc'])
    roc_smote = np.array(res['smote']['roc'])
    
    diff_bal_none = roc_bal - roc_none
    diff_smote_none = roc_smote - roc_none
    diff_bal_smote = roc_bal - roc_smote
    
    t_bal_none, p_bal_none = stats.ttest_rel(roc_bal, roc_none)
    t_smote_none, p_smote_none = stats.ttest_rel(roc_smote, roc_none)
    t_bal_smote, p_bal_smote = stats.ttest_rel(roc_bal, roc_smote)
    
    print('\n--- PAIRED FOLD DIFFERENCES & T-TESTS ---')
    print(f'Fold diffs (Balanced - None): {[round(x, 6) for x in diff_bal_none]}')
    print(f'Mean diff (Balanced - None): {np.mean(diff_bal_none):+.6f}, t-stat={t_bal_none:.4f}, p-val={p_bal_none:.4f}')
    print(f'Fold diffs (SMOTE - None):    {[round(x, 6) for x in diff_smote_none]}')
    print(f'Mean diff (SMOTE - None):    {np.mean(diff_smote_none):+.6f}, t-stat={t_smote_none:.4f}, p-val={p_smote_none:.4f}')
    print(f'Fold diffs (Balanced - SMOTE):{[round(x, 6) for x in diff_bal_smote]}')
    print(f'Mean diff (Balanced - SMOTE):{np.mean(diff_bal_smote):+.6f}, t-stat={t_bal_smote:.4f}, p-val={p_bal_smote:.4f}')
