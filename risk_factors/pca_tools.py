"""
Инструменты сокращения размерности: PCA и факторный анализ.

Две основные задачи:
  * PCA кривой доходности по ИЗМЕНЕНИЯМ ставок (ковариационная матрица, одни единицы
    измерения, б.п.): первые 3 компоненты традиционно интерпретируются как
    уровень, наклон, кривизна;
  * PCA/факторный анализ доходностей акций по КОРРЕЛЯЦИЯМ (ряды стандартизуются,
    т.к. волатильности разные): первая компонента это рыночный фактор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis


def pca_on_changes(changes: pd.DataFrame, n_components: int = 3,
                   standardize: bool = False):
    """
    PCA для матрицы приращений (например, изменений доходностей кривой).
    standardize=False, работаем по ковариации (рекомендуется для кривой,
    единицы одинаковые). Возвращает (модель, факторы-во-времени, нагрузки, доля дисперсии).
    """
    X = changes.dropna()
    Xc = (X - X.mean())
    if standardize:
        Xc = Xc / X.std(ddof=0)
    model = PCA(n_components=n_components)
    scores = model.fit_transform(Xc.values)
    scores = pd.DataFrame(scores, index=X.index,
                          columns=[f"PC{i+1}" for i in range(n_components)])
    loadings = pd.DataFrame(model.components_.T, index=changes.columns,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    evr = pd.Series(model.explained_variance_ratio_,
                    index=[f"PC{i+1}" for i in range(n_components)], name="explained_var")
    return model, scores, loadings, evr


def full_scree(changes: pd.DataFrame, standardize: bool = False) -> pd.Series:
    """Доля объяснённой дисперсии по ВСЕМ компонентам (для scree-графика и выбора k)."""
    X = changes.dropna()
    Xc = X - X.mean()
    if standardize:
        Xc = Xc / X.std(ddof=0)
    model = PCA().fit(Xc.values)
    return pd.Series(model.explained_variance_ratio_,
                     index=[f"PC{i+1}" for i in range(len(model.explained_variance_ratio_))],
                     name="explained_var")


def interpret_curve_pcs(loadings: pd.DataFrame) -> pd.DataFrame:
    """
    Помечает первые три компоненты кривой именами level/slope/curvature на основе
    знаковой структуры нагрузок (для подписи и проверки экономического смысла).
    """
    names = {}
    for pc in loadings.columns[:3]:
        v = loadings[pc].values
        same_sign = np.all(v > 0) or np.all(v < 0)
        sign_changes = np.sum(np.diff(np.sign(v)) != 0)
        if same_sign:
            names[pc] = "level (уровень)"
        elif sign_changes == 1:
            names[pc] = "slope (наклон)"
        else:
            names[pc] = "curvature (кривизна)"
    return pd.Series(names, name="interpretation").to_frame()


def factor_analysis(returns: pd.DataFrame, n_factors: int = 3):
    """
    Факторный анализ доходностей (альтернатива PCA). Ряды стандартизуются.
    Возвращает (нагрузки, факторы-во-времени).
    """
    X = returns.dropna()
    Z = (X - X.mean()) / X.std(ddof=0)
    fa = FactorAnalysis(n_components=n_factors, random_state=0)
    scores = fa.fit_transform(Z.values)
    loadings = pd.DataFrame(fa.components_.T, index=returns.columns,
                            columns=[f"F{i+1}" for i in range(n_factors)])
    scores = pd.DataFrame(scores, index=X.index,
                          columns=[f"F{i+1}" for i in range(n_factors)])
    return loadings, scores
