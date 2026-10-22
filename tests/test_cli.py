from traj_opt.cli import main


def test_assess_and_sweep_smoke(capsys):
    main(["assess", "--seeds", "2", "--trials", "20", "--jobs", "2"])
    out = capsys.readouterr().out
    assert "P(loss of separation)" in out and "placeholders" in out
    main(["sweep", "--seeds", "2", "--trials", "20", "--jobs", "2", "--scales", "1,3"])
    out = capsys.readouterr().out
    assert "envelope" in out and "turn_xtrack_sd" in out
