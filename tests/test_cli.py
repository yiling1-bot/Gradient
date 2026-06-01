from gradient_optimizer.cli import main


def test_cli_runs_quadratic_1d(capsys):
    exit_code = main(["quadratic-1d", "--max-iterations", "200"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "point=" in output
    assert "value=" in output
