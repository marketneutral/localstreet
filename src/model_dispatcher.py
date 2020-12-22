import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()
log_reg = LogisticRegression(
    C=0.1,
    max_iter=1000,
    tol=0.1,
    verbose=10,
    penalty='l1',
    solver='liblinear',
    random_state=42
)

lr_pipe = Pipeline(steps=[
    ('imputer', imp_mean),
    ('scaler', scaler),
    ('log_reg', log_reg)
])

models = {
    'baseline_log_reg': lr_pipe
}
    
