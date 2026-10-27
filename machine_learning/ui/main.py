"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    # Load style sheet if present
    try:
        with open("styles.qss", "r", encoding="utf-8") as fh:
            app.setStyleSheet(fh.read())
    except FileNotFoundError:
        pass

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
