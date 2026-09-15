from scripts.data_collector.yahoo import collector as yahoo_collector


def test_yahoo_collector_can_skip_index_download(monkeypatch):
    calls = []
    monkeypatch.setattr(yahoo_collector.BaseCollector, "collector_data", lambda self: calls.append("quotes"))

    run = object.__new__(yahoo_collector.YahooCollectorCN1d)
    run.download_index = False
    monkeypatch.setattr(run, "download_index_data", lambda: calls.append("index"))
    run.collector_data()

    assert calls == ["quotes"]


def test_yahoo_collector_keeps_index_download_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(yahoo_collector.BaseCollector, "collector_data", lambda self: calls.append("quotes"))

    run = object.__new__(yahoo_collector.YahooCollectorCN1d)
    run.download_index = True
    monkeypatch.setattr(run, "download_index_data", lambda: calls.append("index"))
    run.collector_data()

    assert calls == ["quotes", "index"]
