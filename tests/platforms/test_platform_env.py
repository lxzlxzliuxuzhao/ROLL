from roll.platforms.platform import Platform


def test_common_envs_default_pythonhashseed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    assert Platform.get_common_envs()["PYTHONHASHSEED"] == "0"


def test_common_envs_preserve_pythonhashseed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "123")

    assert Platform.get_common_envs()["PYTHONHASHSEED"] == "123"
