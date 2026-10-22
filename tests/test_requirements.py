import math

import pytest

from traj_opt.assess import binom_upper
from traj_opt.model import Params
from traj_opt.requirements import load


def test_repo_requirements_load():
    req = load("requirements.toml")
    assert req.params.sep_h == 50.0 and req.disp.gust_sd > 0 and req.budget.p_los_max == 1e-4
    assert set(req.placeholders) == set(req.sources)


def test_params_base_kept():
    assert load("requirements.toml", Params(dx=10.0)).params.dx == 10.0


@pytest.mark.parametrize(
    "body",
    [
        'sep_h = { value = 1.0, source = "guess" }',
        'not_a_field = { value = 1.0, source = "spec" }',
    ],
)
def test_bad_entries_rejected(tmp_path, body):
    f = tmp_path / "r.toml"
    f.write_text("[s]\n" + body + "\n")
    with pytest.raises(ValueError):
        load(str(f))


@pytest.mark.parametrize("n", [10, 300, 30_000])
def test_binom_upper_zero_events(n):
    assert binom_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-6)


@pytest.mark.parametrize("k,n", [(1, 50), (5, 100), (40, 1000)])
def test_binom_upper_is_the_95pct_bound(k, n):
    p = binom_upper(k, n)
    cdf = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert cdf == pytest.approx(0.05, abs=1e-6) and p > k / n


def test_binom_upper_edges():
    assert binom_upper(3, 3) == 1.0 and binom_upper(0, 0) == 1.0
