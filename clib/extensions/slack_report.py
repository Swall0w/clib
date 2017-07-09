import sys
from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module
import requests
import json


class SlackReport(extension.Extension):
    def __init__(self, entries, log_report='LogReport',
                 username="", url="", channel="", out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._log_len = 0  # number of observations already printed
        self._out = out

        # format information

        entry_widths = [max(10, len(s)) for s in entries]
        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._header = header  # printed at the first call

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, '  ' * (w + 2)))
        self._templates = templates
        self.username = username
        self.url = url
        self.channel = channel

    def __call__(self, trainer):
        if self._header:
            self._throw_slack(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError(
                'log report has a wrong type {0}'.format(type(log_report)))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._observation_throw_slack(log[log_len])
            log_len += 1
        self._log_len = log_len

    def _throw_slack(self, text):
        try:
            payload_dic = {
                "text": text,
                "username": self.username,
                "channel": self.channel,
            }
            requests.post(self.url, data=json.dumps(payload_dic))
        except:
            self._out.write("error!")

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _observation_throw_slack(self, observation):
        text = ""
        for entry, template, empty in self._templates:
            if entry in observation:
                text += template.format(observation[entry])
            else:
                text += empty
        self._throw_slack(text)
