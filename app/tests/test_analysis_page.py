from app.ui.analysis_page import AnalysisPage


def test_export_button_visible(qapp):
    page = AnalysisPage()
    page.show()
    qapp.processEvents()
    btn = page.get_export_button()
    assert btn.text() == "Export Report"
    assert btn.isVisible()


def test_export_button_centered(qapp):
    page = AnalysisPage()
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    btn = page.get_export_button()
    page_center = page.rect().center().x()
    btn_center = btn.mapTo(page, btn.rect().center()).x()
    assert abs(btn_center - page_center) <= 2


def test_set_results_stores_metadata(qapp):
    page = AnalysisPage()
    results = {
        "classifications": {},
        "localizations": [],
        "transcription": {"text": "hi", "words": [], "duration_sec": 0.5},
    }
    page.set_results(results, audio=None, sample_rate=16000, language="hindi", filename="clip.wav")
    assert page._results_data is results
    assert page._filename == "clip.wav"
    assert page._language == "hindi"
    assert page._sample_rate == 16000
