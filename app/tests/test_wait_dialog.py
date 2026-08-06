from PySide6.QtWidgets import QLabel

from app.ui.wait_dialog import WaitDialog


def test_wait_dialog_message_and_finish(qapp):
    dialog = WaitDialog()
    label = dialog.findChild(QLabel)
    assert label is not None
    assert "Generating transcript" in label.text()
    dialog.show()
    qapp.processEvents()
    assert dialog.isVisible()
    dialog.finish()
    assert not dialog.isVisible()
