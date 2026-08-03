from app.ui.home_page import HomePage


def test_home_page_has_tabs_and_panel(qapp):
    page = HomePage()
    tabs = page._tabs
    assert tabs.count() == 2
    assert tabs.tabText(0) == "Passage"
    assert tabs.tabText(1) == "Files"
    assert page.get_file_panel() is tabs.widget(1)


def test_home_page_reemits_file_selected(qapp, tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x")
    page = HomePage()
    emitted = []
    page.file_selected.connect(emitted.append)
    page.get_file_panel().add_recent(str(a))
    page.get_file_panel()._on_recent_activated(page.get_file_panel()._recent_list.item(0))
    assert emitted == [str(a)]
