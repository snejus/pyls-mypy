# -*- coding: utf-8 -*-
"""
File that contains the python-lsp-server plugin pylsp-mypy.

Created on Fri Jul 10 09:53:57 2020

@author: Richard Kellnberger
"""

import ast
import atexit
import collections
import logging
import os
import os.path
import re
import shutil
import subprocess
from configparser import ConfigParser
from functools import partial
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, TypedDict

import psutil
from typing_extensions import NotRequired

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found]

from mypy import api as mypy_api
from pylsp import hookimpl
from pylsp.config.config import Config
from pylsp.workspace import Document, Workspace

line_pattern = re.compile(
    r"^(?P<file>.+):(?P<start_line>\d+):(?P<start_col>\d*):(?P<end_line>\d*):(?P<end_col>\d*): "
    r"(?P<severity>\w+): (?P<message>.+?)(?: +\[(?P<code>.+)\])?$"
)

whole_line_pattern = re.compile(  # certain mypy warnings do not report start-end ranges
    r"^(?P<file>.+?):(?P<start_line>\d+):(?:(?P<start_col>\d+):)? "
    r"(?P<severity>\w+): (?P<message>.+?)(?: +\[(?P<code>.+)\])?$"
)

log = logging.getLogger(__name__)


class LineRange(TypedDict):
    """LineRange TypedDict for mypy diagnostics."""

    line: int
    character: int


class Range(TypedDict):
    """Range TypedDict for mypy diagnostics."""

    start: LineRange
    end: LineRange


class Diagnostic(TypedDict):
    """Diagnostic TypedDict for mypy diagnostics."""

    source: Literal["mypy"]
    range: Range
    message: str
    severity: int
    code: NotRequired[str]


# A mapping from workspace path to config file path
mypyConfigFileMap: Dict[str, Optional[str]] = {}

settingsCache: Dict[str, Dict[str, Any]] = {}

tmpFile: Optional[IO[bytes]] = None

# In non-live-mode the file contents aren't updated.
# Returning an empty diagnostic clears the diagnostic result,
# so store a cache of last diagnostics for each file a-la the pylint plugin,
# so we can return some potentially-stale diagnostics.
# https://github.com/python-lsp/python-lsp-server/blob/v1.0.1/pylsp/plugins/pylint_lint.py#L55-L62
last_diagnostics: Dict[tuple[str, str], List[Diagnostic]] = collections.defaultdict(
    list
)

# Windows started opening opening a cmd-like window for every subprocess call
# This flag prevents that.
# This flag is new in python 3.7
# This flag only exists on Windows
windows_flag: Dict[str, int] = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}  # type: ignore
)

_attr_prefixes = [
    "Skipping analyzing",
    "Signature of",
    "Name",
    "keyword argument",
    "attribute",
    "for",
    "Cannot resolve keyword",
]
FIND_ATTR_SUFFIX = re.compile(
    rf'({"|".join(_attr_prefixes)}) [\'"]([^"\']+)[\'"]'
).search
FIND_ATTR_PREFIX = re.compile(r'( )?"([^"]+)" undefined').search


def parse_line(line: str, document: Document) -> Optional[Diagnostic]:
    """Return a language-server diagnostic from a line of the Mypy error report.

    optionally, use the whole document to provide more context on it.


    Parameters
    ----------
    line : str
        Line of mypy output to be analysed.
    document : Optional[Document], optional
        Document in wich the line is found. The default is None.

    Returns
    -------
    Optional[Dict[str, Any]]
        The dict with the lint data.

    """
    line_match = line_pattern.match(line) or whole_line_pattern.match(line)

    if not line_match:
        return None

    result = line_match.groupdict()

    code = result["code"]
    message = result["message"]
    if not code and "Revealed" not in message:
        return None

    severity = result["severity"]
    if code in {"no-untyped-call", "no-untyped-def", "type-arg"}:
        severity = "note"
        if "def test_" in line:
            return None

    lineno = int(result["start_line"]) - 1  # 0-based line number
    offset = int(result.get("start_col") or 1) - 1  # 0-based offset
    end_lineno = int(result.get("end_line") or lineno + 1) - 1
    end_offset = int(result.get("end_col") or 1)  # end is exclusive

    if (
        code in {None, "misc", "django-manager-missing", "explicit-override"}
        or (code == "return" and "Missing" in message)
        or (code == "override" and "incompatible with supertype" in message)
    ):
        end_lineno = lineno
        end_offset = offset + 1

    # Heuristic: If the message mentions a quoted symbol (attribute / argument / name),
    # try to narrow the diagnostic range exactly to that symbol within the original
    # source slice we already calculated. Improves UX vs highlighting whole line.
    if matches := FIND_ATTR_SUFFIX(message) or FIND_ATTR_PREFIX(message):
        type_, prop = matches.groups()
        for lineno, this_line in enumerate(
            document.lines[lineno : end_lineno + 1], lineno
        ):
            if m := re.search(
                rf"\.{prop}\b" if type_ == "attribute" else prop, this_line
            ):
                offset, end_offset = m.span(0)
                end_lineno = lineno
                break

    return {
        "source": "mypy",
        "range": {
            "start": {"line": lineno, "character": offset},
            "end": {"line": end_lineno, "character": end_offset},
        },
        "message": message,
        "severity": {"error": 1, "warning": 2, "note": 3}[severity],
        "code": code,
    }


def didSettingsChange(workspace: str, settings: Dict[str, Any]) -> None:
    """Handle relevant changes to the settings between runs."""
    configSubPaths = settings.get("config_sub_paths", [])
    if settingsCache[workspace].get("config_sub_paths", []) != configSubPaths:
        mypyConfigFile = findConfigFile(
            workspace,
            configSubPaths,
            ["mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg"],
            True,
        )
        mypyConfigFileMap[workspace] = mypyConfigFile
        settingsCache[workspace] = settings.copy()


def match_exclude_patterns(document_path: str, exclude_patterns: list) -> bool:
    """Check if the current document path matches any of the configures exlude patterns."""
    document_path = document_path.replace(os.sep, "/")

    for pattern in exclude_patterns:
        try:
            if re.search(pattern, document_path):
                log.debug(f"{document_path} matches exclude pattern '{pattern}'")
                return True
        except re.error as e:
            log.error(f"pattern {pattern} is not a valid regular expression: {e}")

    return False


def get_cmd(settings: Dict[str, Any], cmd: str) -> List[str]:
    """
    Get the command to run from settings, falling back to searching the PATH.
    If the command is not found in the settings and is not available on the PATH, an
    empty list is returned.
    """
    command_key = f"{cmd}_command"
    command: List[str] = settings.get(command_key, [])

    if not (command and os.getenv("PYLSP_MYPY_ALLOW_DANGEROUS_CODE_EXECUTION")):
        # env var is required to allow command from settings
        if shutil.which(cmd):  # Fallback to PATH
            log.debug(
                f"'{command_key}' not found in settings or not allowed, using '{cmd}' from PATH"
            )
            command = [cmd]
        else:  # Fallback to API
            command = []

    log.debug(f"Using {cmd} command: {command}")

    return command


@hookimpl
def pylsp_lint(
    config: Config, workspace: Workspace, document: Document, is_saved: bool
) -> List[Diagnostic]:
    """
    Call the linter.

    Parameters
    ----------
    config : Config
        The pylsp config.
    workspace : Workspace
        The pylsp workspace.
    document : Document
        The document to be linted.
    is_saved : bool
        Weather the document is saved.

    Returns
    -------
    List[Dict[str, Any]]
        List of the linting data.

    """
    settings = config.plugin_settings("pylsp_mypy")
    oldSettings1 = config.plugin_settings("mypy-ls")
    oldSettings2 = config.plugin_settings("mypy_ls")
    if oldSettings1 != {} or oldSettings2 != {}:
        raise NameError(
            "Your configuration uses an old namespace (mypy-ls or mypy_ls)."
            + "This should be changed to pylsp_mypy"
        )
    if settings == {}:
        settings = oldSettings1
        if settings == {}:
            settings = oldSettings2

    didSettingsChange(workspace.root_path, settings)

    # Running mypy with a single file (document) ignores any exclude pattern
    # configured with mypy. We can now add our own exclude section like so:
    # [tool.pylsp-mypy]
    # exclude = ["tests/*"]
    exclude_patterns = settings.get("exclude", [])

    if match_exclude_patterns(
        document_path=document.path, exclude_patterns=exclude_patterns
    ):
        log.debug(
            f"Not running because {document.path} matches "
            f"exclude patterns '{exclude_patterns}'"
        )
        return []

    if settings.get("report_progress", False):
        with workspace.report_progress("lint: mypy"):
            return get_diagnostics(workspace, document, settings, is_saved)
    else:
        return get_diagnostics(workspace, document, settings, is_saved)


def get_diagnostics_key(document: Document) -> tuple[str, str]:
    root_dir_path = Path().absolute()
    filepath = Path(document.path).resolve()
    return str(filepath.relative_to(root_dir_path)), document.source


def get_diagnostics(
    workspace: Workspace,
    document: Document,
    settings: Dict[str, Any],
    is_saved: bool,
) -> List[Diagnostic]:
    """
    Lints.

    Parameters
    ----------
    workspace : Workspace
        The pylsp workspace.
    document : Document
        The document to be linted.
    is_saved : bool
        Weather the document is saved.

    Returns
    -------
    List[Dict[str, Any]]
        List of the linting data.

    """
    log.warning(
        "lint settings = %s document.path = %s is_saved = %s",
        settings,
        document.path,
        is_saved,
    )

    dmypy = settings.get("dmypy", False)

    args = ["--show-error-end", "--no-pretty"]

    root_dir = str(Path().absolute())
    try:
        diagnostics_key = get_diagnostics_key(document)
    except ValueError:
        return []

    file, _ = diagnostics_key

    def get_cached_diagnostics() -> list[Diagnostic]:
        if diagnostics := last_diagnostics[diagnostics_key]:
            # On-launch the document isn't marked as saved, so fall through and run
            # the diagnostics anyway even if the file contents may be out of date.
            log.warning(
                "Returning cached diagnostics len(cached) = %s",
                len(diagnostics),
            )
            return diagnostics
        log.warning("Removing diagnostics")
        return []

    if not is_saved:
        log.warning("Unsaved file")
        return get_cached_diagnostics()

    mypyConfigFile = mypyConfigFileMap.get(workspace.root_path)
    if mypyConfigFile:
        args.extend(["--config-file", mypyConfigFile])

    if not dmypy:
        args.append(file)

    if settings.get("strict", False):
        args.append("--strict")

    args.extend(settings.get("overrides", []))
    exit_status = 0

    run = partial(subprocess.run, capture_output=True, **windows_flag, encoding="utf-8")

    if not dmypy:
        args = [
            "--incremental",
            "--follow-imports",
            settings.get("follow-imports", "silent"),
            "--cache-fine-grained",
            "--cache-dir=/home/sarunas/.cache/mypy",
            *args,
        ]

        mypy_command: List[str] = get_cmd(settings, "mypy")

        if mypy_command:
            # mypy exists on path
            # -> use mypy on path
            log.warning("executing mypy on path: %s", " ".join(args))
            response = run([*mypy_command, *args])
            report = response.stdout
            errors = response.stderr
            exit_status = response.returncode
        else:
            # mypy does not exist on PATH and was not provided by settings,
            # but must exist in the env pylsp-mypy is installed in
            # -> use mypy via api
            log.warning("executing mypy via API: %s", " ".join(args))
            report, errors, exit_status = mypy_api.run(args)
    else:
        # If dmypy daemon is non-responsive calls to run will block.
        # Check daemon status, if non-zero daemon is dead or hung.
        # If daemon is hung, kill will reset
        # If daemon is dead/absent, kill will no-op.
        # In either case, reset to fresh state

        if dmypy_command := get_cmd(settings, "dmypy"):

            def run_dmypy(*args, **kwargs) -> tuple[str, str, int]:
                dmypy_args = [
                    *dmypy_command,
                    "--status-file",
                    settings.get("dmypy_status_file", ".dmypy.json"),
                    *args,
                ]
                log.warning("dmypy run args = %s via path", dmypy_args)
                result = run(dmypy_args, **kwargs)
                return result.stdout, result.stderr, result.returncode

            @atexit.register
            def kill_dmypy() -> None:
                run_dmypy("kill")

            started, checking = False, False
            for process in psutil.process_iter(["name", "cmdline"]):
                if process.name() == "dmypy":
                    cmdline = set(process.cmdline())
                    if root_dir in cmdline:
                        started = True
                    if {
                        document.path,
                        tmpFile.name if tmpFile else document.path,
                    } & cmdline:
                        checking = True

            if not started:
                run_dmypy("start", "--", root_dir, *args)

            if not checking:
                report, errors, exit_status = run_dmypy("check", file)
                if exit_status > 1:
                    report, errors, exit_status = run_dmypy("check", file)

    log.debug("report:\n%s", report)
    log.debug("errors:\n%s", errors)

    diagnostics: list[Diagnostic] = []

    # Expose generic mypy error on the first line.
    if errors:
        diagnostics.append(
            {
                "source": "mypy",
                "range": {
                    "start": {"line": 0, "character": 0},
                    # Client is supposed to clip end column to line length.
                    "end": {"line": 0, "character": 1000},
                },
                "message": errors,
                "severity": (
                    1 if exit_status != 0 else 2
                ),  # Error if exited with error or warning.
            }
        )

    for line in report.splitlines():
        if line.startswith(file):
            log.debug("parsing: line = %r", line)
            if diag := parse_line(line, document):
                diagnostics.append(diag)

    log.warning("pylsp-mypy len(diagnostics) = %s", len(diagnostics))

    last_diagnostics[diagnostics_key] = diagnostics
    return diagnostics


@hookimpl
def pylsp_settings(config: Config) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Read the settings.

    Parameters
    ----------
    config : Config
        The pylsp config.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]]
        The config dict.

    """
    configuration = init(config._root_path)
    return {"plugins": {"pylsp_mypy": configuration}}


def init(workspace: str) -> Dict[str, str]:
    """
    Find plugin and mypy config files and creates the temp file should it be used.

    Parameters
    ----------
    workspace : str
        The path to the current workspace.

    Returns
    -------
    Dict[str, str]
        The plugin config dict.

    """
    log.warning("init workspace = %s", workspace)

    configuration = {}
    path = findConfigFile(
        workspace,
        [],
        ["pylsp-mypy.cfg", "mypy-ls.cfg", "mypy_ls.cfg", "pyproject.toml"],
        False,
    )
    if path:
        if "pyproject.toml" in path:
            with open(path, "rb") as file:
                configuration = tomllib.load(file).get("tool").get("pylsp-mypy")
        else:
            with open(path) as file:
                configuration = ast.literal_eval(file.read())

    configSubPaths = configuration.get("config_sub_paths", [])
    mypyConfigFile = findConfigFile(
        workspace,
        configSubPaths,
        ["mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg"],
        True,
    )
    mypyConfigFileMap[workspace] = mypyConfigFile
    settingsCache[workspace] = configuration.copy()

    log.warning("mypyConfigFile = %s configuration = %s", mypyConfigFile, configuration)
    return configuration


def findConfigFile(
    path: str, configSubPaths: List[str], names: List[str], mypy: bool
) -> Optional[str]:
    """
    Search for a config file.

    Search for a file of a given name from the directory specifyed by path through all parent
    directories. The first file found is selected.

    Parameters
    ----------
    path : str
        The path where the search starts.
    configSubPaths : List[str]
        Additional sub search paths in which mypy configs might be located
    names : List[str]
        The file to be found (or alternative names).
    mypy : bool
        whether the config file searched is for mypy (plugin otherwise)

    Returns
    -------
    Optional[str]
        The path where the file has been found or None if no matching file has been found.

    """
    start = Path(path).joinpath(names[0])  # the join causes the parents to include path
    for parent in start.parents:
        for name in names:
            for subPath in [""] + configSubPaths:
                file = parent.joinpath(subPath).joinpath(name)
                if file.is_file():
                    if file.name in ["mypy-ls.cfg", "mypy_ls.cfg"]:
                        raise NameError(
                            f"{str(file)}: {file.name} is no longer supported, you should rename "
                            "your config file to pylsp-mypy.cfg or preferably use a pyproject.toml "
                            "instead."
                        )
                    if file.name == "pyproject.toml":
                        with open(file, "rb") as fileO:
                            configPresent = (
                                tomllib.load(fileO)
                                .get("tool", {})
                                .get("mypy" if mypy else "pylsp-mypy")
                                is not None
                            )
                        if not configPresent:
                            continue
                    if file.name == "setup.cfg":
                        config = ConfigParser()
                        config.read(str(file))
                        if "mypy" not in config:
                            continue
                    return str(file)
    # No config file found in the whole directory tree
    # -> check mypy default locations for mypy config
    if mypy:
        defaultPaths = ["~/.config/mypy/config", "~/.mypy.ini"]
        XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME")
        if XDG_CONFIG_HOME:
            defaultPaths.insert(0, f"{XDG_CONFIG_HOME}/mypy/config")
        for path in defaultPaths:
            if Path(path).expanduser().exists():
                return str(Path(path).expanduser())
    return None


@hookimpl
def pylsp_code_actions(
    config: Config,
    workspace: Workspace,
    document: Document,
    range: Dict,
    context: Dict,
) -> List[Dict]:
    """
    Provide code actions to ignore errors.

    Parameters
    ----------
    config : pylsp.config.config.Config
        Current config.
    workspace : pylsp.workspace.Workspace
        Current workspace.
    document : pylsp.workspace.Document
        Document to apply code actions on.
    range : Dict
        Range argument given by pylsp.
    context : Dict
        CodeActionContext given as dict.

    Returns
    -------
      List of dicts containing the code actions.
    """
    actions = []
    # Code actions based on diagnostics
    for diagnostic in context.get("diagnostics", []):
        if diagnostic["source"] != "mypy":
            continue
        if "code" not in diagnostic:
            continue
        code = diagnostic["code"]
        lineNumberEnd = diagnostic["range"]["end"]["line"]
        line = document.lines[lineNumberEnd]
        endOfLine = len(line) - 1
        start = {"line": lineNumberEnd, "character": endOfLine}
        edit_range = {"start": start, "end": start}
        edit = {"range": edit_range, "newText": f"  # type: ignore[{code}]"}

        action = {
            "title": f"# type: ignore[{code}]",
            "kind": "quickfix",
            "diagnostics": [diagnostic],
            "edit": {"changes": {document.uri: [edit]}},
        }
        actions.append(action)
    if context.get("diagnostics", []) != []:
        return actions

    # Code actions based on current selected range
    for diagnostic in last_diagnostics[get_diagnostics_key(document)]:
        lineNumberStart = diagnostic["range"]["start"]["line"]
        lineNumberEnd = diagnostic["range"]["end"]["line"]
        rStart = range["start"]["line"]
        rEnd = range["end"]["line"]
        if (rStart <= lineNumberStart and rEnd >= lineNumberStart) or (
            rStart <= lineNumberEnd and rEnd >= lineNumberEnd
        ):
            code = diagnostic["code"]
            line = document.lines[lineNumberEnd]
            endOfLine = len(line) - 1
            start = {"line": lineNumberEnd, "character": endOfLine}
            edit_range = {"start": start, "end": start}
            edit = {"range": edit_range, "newText": f"  # type: ignore[{code}]"}
            action = {
                "title": f"# type: ignore[{code}]",
                "kind": "quickfix",
                "edit": {"changes": {document.uri: [edit]}},
            }
            actions.append(action)

    return actions


@atexit.register
def close() -> None:
    """
    Deltes the tempFile should it exist.

    Returns
    -------
    None.

    """
    if tmpFile and tmpFile.name:
        os.unlink(tmpFile.name)
