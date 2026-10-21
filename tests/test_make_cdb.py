import json
import shlex

import pytest

from make_cdb import convert_entry, main


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    prefix = tmp_path / "environment with spaces"
    for path in (
        "include",
        "bin",
        "conda-meta",
        "aarch64-conda-linux-gnu/sysroot",
        "targets/aarch64-linux/include",
        "lib/gcc/aarch64-conda-linux-gnu/14.4.0",
        "lib/gcc/aarch64-conda-linux-gnu/15.3.0",
    ):
        (prefix / path).mkdir(parents=True, exist_ok=True)
    (prefix / "lib/gcc/aarch64-conda-linux-gnu/15.3.0/crtbegin.o").touch()
    (prefix / "targets/aarch64-linux/include/cuda.h").touch()
    return prefix


@pytest.fixture
def entry(prefix, tmp_path):
    return {
        "directory": str(tmp_path / "cdb/build"),
        "file": "source file.cu",
        "output": "source file.o",
        "arguments": [str(prefix / "bin/nvcc"), "-c", "source file.cu"],
    }


@pytest.mark.parametrize("form", ["command", "arguments", "both"])
def test_conversion_preserves_command_and_metadata(prefix, entry, form):
    arguments = entry["arguments"] + [
        '-DNAME="a b && c"',
        "-gencode=arch=compute_100a,code=[sm_100a]",
        "-Xcompiler=-fPIC,-Wall",
        "--default-stream",
        "per-thread",
        "-x",
        "cu",
        "--options-file",
        "includes.rsp",
        "@flags.rsp",
    ]
    entry["arguments"] = arguments.copy()
    if form == "command":
        entry["command"] = shlex.join(entry.pop("arguments"))
    elif form == "both":
        entry["command"] = 'ignored invalid "command'
    original = json.dumps(entry)

    converted = convert_entry(entry)
    args = converted.pop("arguments")
    assert converted == {
        k: v for k, v in entry.items() if k not in ("command", "arguments")
    }
    assert args[0] == arguments[0]
    assert args[-len(arguments) + 1 :] == arguments[1:]
    assert {
        str(prefix / "include"),
        f"--gcc-install-dir={prefix}/lib/gcc/aarch64-conda-linux-gnu/15.3.0",
        "--target=aarch64-conda-linux-gnu",
        f"--sysroot={prefix}/aarch64-conda-linux-gnu/sysroot",
        f"--cuda-path={prefix}/targets/aarch64-linux",
        "--cuda-gpu-arch=sm_100a",
    } <= set(args)
    converted["arguments"] = args
    assert convert_entry(converted) == converted
    assert json.dumps(entry) == original


@pytest.mark.parametrize("source", ["compiler", "active", "explicit", "none"])
def test_prefix_selection(prefix, entry, monkeypatch, source):
    entry["arguments"][0] = str(prefix / "bin/cc") if source == "compiler" else "cc"
    options = {}
    if source == "active":
        monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    elif source in ("compiler", "explicit"):
        monkeypatch.setenv("CONDA_PREFIX", "/unrelated/environment")
        if source == "explicit":
            options["prefix"] = prefix

    converted = convert_entry(entry, **options)
    if source == "none":
        assert converted == entry
    else:
        assert str(prefix / "include") in converted["arguments"]
    assert not any(a.startswith("--cuda") for a in converted["arguments"])


@pytest.mark.parametrize("gcc", ["installed", "missing", "multiple"])
def test_gcc_discovery(prefix, entry, gcc):
    entry["arguments"][0] = str(prefix / "bin/cc")
    installations = prefix / "lib/gcc/aarch64-conda-linux-gnu"
    if gcc == "missing":
        (installations / "15.3.0/crtbegin.o").unlink()
    elif gcc == "multiple":
        (installations / "14.4.0/crtbegin.o").touch()
        with pytest.raises(ValueError, match="Multiple GCC"):
            convert_entry(entry)
        entry["arguments"].append(f"--gcc-install-dir={installations}/15.3.0")

    args = convert_entry(entry)["arguments"]
    assert str(prefix / "include") in args
    assert (f"--gcc-install-dir={installations}/15.3.0" in args) == (gcc != "missing")


@pytest.mark.parametrize("layout", ["split", "traditional"])
def test_cuda_layout(prefix, entry, layout):
    if layout == "traditional":
        (prefix / "include/cuda.h").touch()
    root = prefix if layout == "traditional" else prefix / "targets/aarch64-linux"
    assert f"--cuda-path={root}" in convert_entry(entry)["arguments"]


@pytest.mark.parametrize(
    "flags,override,expected",
    [
        (["-gencode", "arch=compute_90,code=[compute_90,sm_90]"], None, "sm_90"),
        (["--gpu-architecture=compute_100f"], None, "sm_100f"),
        (["-arch", "sm_89"], None, "sm_89"),
        (["--gpu-code", "sm_80,sm_90"], None, "sm_80"),
        ([], "sm_90", "sm_90"),
    ],
)
def test_cuda_architecture(entry, flags, override, expected):
    entry["arguments"].extend(flags)
    assert (
        f"--cuda-gpu-arch={expected}"
        in convert_entry(entry, cuda_gpu_arch=override)["arguments"]
    )


def test_explicit_flags_take_precedence(prefix, entry):
    entry["arguments"] += [
        "-isystem",
        str(prefix / "include"),
        "--target=custom",
        "--sysroot=/custom",
        "--gcc-install-dir=/custom/gcc",
        "--cuda-path=/custom/cuda",
        "--cuda-gpu-arch=sm_90",
    ]
    assert convert_entry(entry) == entry


def test_cli_conversion(entry, tmp_path, capsys):
    source = tmp_path / "source.json"
    destination = tmp_path / "clangd/compile_commands.json"
    source.write_text(json.dumps([entry]))
    before = source.read_bytes()
    main([str(source), str(destination)])
    converted = json.loads(destination.read_text())
    assert len(converted) == 1
    assert converted[0]["file"] == entry["file"]
    assert "--target=aarch64-conda-linux-gnu" in converted[0]["arguments"]
    assert source.read_bytes() == before
    assert f"Wrote 1 entries to {destination}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case", ["no_paths", "source_only", "same", "symlink", "hardlink"]
)
def test_cli_requires_distinct_paths(tmp_path, monkeypatch, capsys, case):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "compile_commands.json"
    source.write_text("[]")
    destination = source
    if case in ("symlink", "hardlink"):
        destination = tmp_path / "alias.json"
        if case == "symlink":
            destination.symlink_to(source)
        else:
            destination.hardlink_to(source)
    paths = [] if case == "no_paths" else [str(source)]
    if case not in ("no_paths", "source_only"):
        paths.append(str(destination))

    with pytest.raises(SystemExit) as error:
        main(paths)
    assert error.value.code == 2
    assert source.read_text() == "[]"
    message = capsys.readouterr().err
    assert (
        "required" in message
        if case in ("no_paths", "source_only")
        else "different files" in message
    )


@pytest.mark.parametrize(
    "case", ["not_array", "invalid_entry", "pipeline", "partial", "empty_arguments"]
)
def test_cli_invalid_input_preserves_database(entry, tmp_path, capsys, case):
    invalid = {
        "not_array": {},
        "invalid_entry": [None],
        "pipeline": [
            {**entry, "arguments": entry["arguments"] + ["&&", "echo", "wrong"]}
        ],
        "partial": [entry, {}],
        "empty_arguments": [{**entry, "arguments": [], "command": "nvcc -c source.cu"}],
    }[case]
    source = tmp_path / "compile_commands.json"
    destination = tmp_path / "output.json"
    source.write_text(json.dumps(invalid))
    destination.write_text("existing destination")
    before = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        main([str(source), str(destination)])
    assert error.value.code == 2
    assert source.read_bytes() == before
    assert destination.read_text() == "existing destination"
    assert "error:" in capsys.readouterr().err
