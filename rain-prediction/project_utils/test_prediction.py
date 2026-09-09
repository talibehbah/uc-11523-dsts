import numpy as np

from prediction import predict, rmse


def test_rmse_perfect_match():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y = np.array([0.0, 0.0])
    yhat = np.array([3.0, 4.0])
    assert abs(rmse(y, yhat) - 3.5355339059) < 1e-6


def test_predict_runs(capsys):
    predict()
    captured = capsys.readouterr()
    assert "baseline" in captured.out
