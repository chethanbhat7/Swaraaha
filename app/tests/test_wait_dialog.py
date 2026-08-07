from app.ui.wait_dialog import WaitDialog


def test_wait_dialog_message_and_finish(qapp):
    dialog = WaitDialog()
    assert "Generating transcript" in dialog._label.text()
    dialog.show()
    qapp.processEvents()
    assert dialog.isVisible()
    dialog.finish()
    assert not dialog.isVisible()
