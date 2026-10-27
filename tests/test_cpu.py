from app.config import env_bool, threads_for_percent


def test_empty_percent_uses_every_core():
    assert threads_for_percent(8, None) == 8
    assert threads_for_percent(8, "  ") == 8


def test_percent_rounds_to_a_thread_count():
    assert threads_for_percent(8, "50") == 4
    assert threads_for_percent(8, "100") == 8
    assert threads_for_percent(4, "80") == 3
    assert threads_for_percent(8, "1") == 1


def test_invalid_percent_falls_back_to_all_cores():
    assert threads_for_percent(8, "beaucoup") == 8
    assert threads_for_percent(8, "150") == 8


def test_precise_flag_accepts_true_and_false():
    assert env_bool("true") is True
    assert env_bool("false") is False
    assert env_bool(None) is False
    assert env_bool("peut-etre") is False
