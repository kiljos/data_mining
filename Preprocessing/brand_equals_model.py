from sklearn.base import BaseEstimator, TransformerMixin

class BrandModelFixer(BaseEstimator, TransformerMixin):
    """Setzt model='unknown' und Flag, wenn brand==model (normalisiert)."""
    def _normalize(self, s):
        return s.str.lower().str.replace(r'[^a-z0-9]', '', regex=True).fillna('')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        brand_norm = self._normalize(X['brand'])
        model_norm = self._normalize(X['model'])
        mask = brand_norm == model_norm
        
        X.loc[mask, 'model'] = 'unknown_model'
        X['brand_eq_model']  = mask.astype(int)
        return X