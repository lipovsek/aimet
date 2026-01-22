# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Automatic Post-Training Quantization"""

import abc
from typing import Any, List, Union

import bokeh.model
import bokeh.embed


class Diagnostics:
    """Diagnostics produced by _EvalSession"""

    def __init__(self):
        self._contents: List[_DiagnosticsContent] = []

    def add(self, content: Union[str, bokeh.model.Model]) -> None:
        """
        Add content of diagnostics.
        :param content: Content of diagnostics.
        :return: None
        """
        if isinstance(content, str):
            c = _PlainTextContent(content)
        elif isinstance(content, bokeh.model.Model):
            c = _BokehModelContent(content)
        else:
            raise RuntimeError
        self._contents.append(c)

    def is_empty(self) -> bool:
        """Return True if and only if the diagnostics is empty."""
        return not bool(self._contents)

    def contains_bokeh(self):
        """Return True if and only if the diagnostics contains a bokeh model object."""
        return any(
            isinstance(content, _BokehModelContent) for content in self._contents
        )

    def __bool__(self):
        return not self.is_empty()

    def __iter__(self):
        return iter(self._contents)


class _DiagnosticsContent(abc.ABC):
    """Content of diagnostics."""

    def __init__(self, content: Any):
        self._content = content

    @abc.abstractmethod
    def get_html_elem(self) -> str:
        """
        Render content as an html element.
        :return: Content rendered as an html element.
        """


class _PlainTextContent(_DiagnosticsContent):
    """Content of diagnostics in plain text."""

    def get_html_elem(self) -> str:
        content = self._content.replace("\n", "<br/>\n")
        return f"<div> {content} </div>"


class _BokehModelContent(_DiagnosticsContent):
    """Content of diagnostics as a bokeh model object."""

    def get_html_elem(self) -> str:
        bokeh_model = self._content
        script, div = bokeh.embed.components(bokeh_model)
        return f"{script}{div}"
