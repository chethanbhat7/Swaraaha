from app.ui.home_page import HomePage


def test_home_page_has_segmented_nav(qapp):
    page = HomePage()
    assert page._passage_btn.text() == "Passage"
    assert page._files_btn.text() == "Files"
    assert page._stack.count() == 2
    assert page.get_pdf_viewer() is page._stack.widget(0)
    assert page.get_file_panel() is page._stack.widget(1)
    assert page._stack.currentIndex() == 0


def test_home_page_nav_switches_pages(qapp):
    page = HomePage()
    page._files_btn.click()
    assert page._stack.currentIndex() == 1
    assert page._files_btn.property("cssClass") == "nav_btn_active"
    assert page._passage_btn.property("cssClass") == "nav_btn"
    page._passage_btn.click()
    assert page._stack.currentIndex() == 0
    assert page._passage_btn.property("cssClass") == "nav_btn_active"
    assert page._files_btn.property("cssClass") == "nav_btn"


def test_home_page_transcript_hidden_by_default(qapp):
    page = HomePage()
    assert page.get_transcription_panel().isHidden()
    page.set_transcript_visible(True)
    assert not page.get_transcription_panel().isHidden()


def test_home_page_reemits_file_selected(qapp, tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x")
    page = HomePage()
    emitted = []
    page.file_selected.connect(emitted.append)
    page.get_file_panel().add_recent(str(a))
    page.get_file_panel()._on_recent_activated(page.get_file_panel()._recent_list.item(0))
    assert emitted == [str(a)]
