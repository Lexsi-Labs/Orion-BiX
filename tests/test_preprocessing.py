import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from orion_bix.sklearn.preprocessing import EnsembleGenerator, PreprocessingPipeline

X = np.random.default_rng(0).lognormal(size=(120, 4))
y = (X[:, 0] > 1).astype(int)


@pytest.mark.parametrize("method", ["none", "power", "quantile", "quantile_rtdl", "robust"])
def test_normalization_method(method):
    pipeline = PreprocessingPipeline(normalization_method=method, random_state=0).fit(X)
    assert pipeline.X_transformed_.shape == X.shape
    assert np.isfinite(pipeline.transform(X * 2)).all()


def test_quantile_uses_normal_output():
    pipeline = PreprocessingPipeline(normalization_method="quantile", random_state=0).fit(X)
    assert isinstance(pipeline.normalizer_, QuantileTransformer)
    assert pipeline.normalizer_.output_distribution == "normal"
    assert pipeline.normalizer_.random_state == 0


def test_quantile_rtdl_is_standardized():
    pipeline = PreprocessingPipeline(normalization_method="quantile_rtdl", random_state=0).fit(X)
    assert list(pipeline.normalizer_.named_steps) == ["quantile_rtdl", "std"]
    assert pipeline.normalizer_.named_steps["quantile_rtdl"].random_state == 0


def test_robust_uses_unit_variance():
    pipeline = PreprocessingPipeline(normalization_method="robust", random_state=0).fit(X)
    assert isinstance(pipeline.normalizer_, RobustScaler)
    assert pipeline.normalizer_.unit_variance is True


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        PreprocessingPipeline(normalization_method="other").fit(X)


def test_ensemble_generator_uses_all_methods():
    methods = ["quantile", "quantile_rtdl", "robust"]
    generator = EnsembleGenerator(n_estimators=3, norm_methods=methods, random_state=0).fit(X, y)
    assert generator.transform(X[:5])
