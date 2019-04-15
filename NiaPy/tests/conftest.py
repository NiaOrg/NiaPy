"""Unit tests configuration file."""

import logging


def pytest_configure(config):
    """Disable verbose output when running tests."""
    logging.basicConfig(level=logging.DEBUG)

    terminal = config.pluginmanager.getplugin('terminal')
    base = terminal.TerminalReporter

    class QuietReporter(base):
        """Reporter that only shows dots when running tests."""

        def __init__(self, *args, **kwargs):
            super(QuietReporter, self).__init__(*args, **kwargs)
            self.verbosity = 0
            self.showlongtestinfo = False
            self.showfspath = False

    terminal.TerminalReporter = QuietReporter
