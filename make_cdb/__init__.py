"""Expose Pixi/Conda toolchains to clangd 23 without translating NVCC flags."""

import argparse
import json
import os
import re
import shlex
import tempfile
from pathlib import Path


def has_option(arguments: list[str], *names: str) -> bool:
    return any(a == n or a.startswith(n + "=") for a in arguments for n in names)


def environment_prefix(arguments: list[str], directory: Path) -> Path | None:
    # Prefer the recorded toolchain over an unrelated active environment. Do not
    # resolve compiler symlinks: their location identifies the environment.
    compiler = Path(arguments[0])
    if not compiler.is_absolute():
        compiler = directory / compiler
    prefix = compiler.parent.parent
    if compiler.parent.name == "bin" and (prefix / "conda-meta").is_dir():
        return prefix.resolve()
    active = os.environ.get("CONDA_PREFIX")
    return Path(active).resolve() if active else None


def toolchain_flags(prefix: Path) -> list[str]:
    if not (prefix / "include").is_dir():
        raise ValueError(
            f"Environment include directory does not exist: {prefix / 'include'}"
        )
    flags = ["-isystem", str(prefix / "include")]
    # Pixi upgrades can leave empty directories for older GCC versions.
    installations = sorted(
        p.parent for p in (prefix / "lib/gcc").glob("*/*/crtbegin.o")
    )
    if len(installations) > 1:
        raise ValueError(f"Multiple GCC installations in {prefix}")
    gcc_install_dir = installations[0] if installations else None
    if gcc_install_dir is not None:
        target = gcc_install_dir.parent.name
        flags.append(f"--gcc-install-dir={gcc_install_dir}")
        sysroot = prefix / target / "sysroot"
        if sysroot.is_dir():
            flags.extend([f"--target={target}", f"--sysroot={sysroot}"])
    return flags


def cuda_architecture(arguments: list[str]) -> str | None:
    # clangd drops NVCC's architecture flags. Retain the first real architecture
    # (including suffixes such as sm_100a); fall back to a virtual architecture.
    values = []
    names = (
        "--generate-code",
        "-gencode",
        "--gpu-architecture",
        "-arch",
        "--gpu-code",
        "-code",
    )
    for i, argument in enumerate(arguments):
        if argument in names and i + 1 < len(arguments):
            values.append(arguments[i + 1])
        elif any(argument.startswith(name + "=") for name in names):
            values.append(argument.split("=", 1)[1])
    value = " ".join(values)
    match = re.search(r"\bsm_(\d+[a-z]?)\b", value) or re.search(
        r"\bcompute_(\d+[a-z]?)\b", value
    )
    return f"sm_{match[1]}" if match else None


def convert_entry(
    entry: dict,
    *,
    prefix: Path | None = None,
    cuda_gpu_arch: str | None = None,
) -> dict:
    """Return a new entry; preserve its directory, metadata, compiler and flags."""
    if not isinstance(entry, dict) or not all(
        isinstance(entry.get(k), str) for k in ("directory", "file")
    ):
        raise ValueError("Each entry must have string 'directory' and 'file' fields")
    arguments = entry.get("arguments")
    if arguments is None:
        command = entry.get("command")
        if not isinstance(command, str):
            raise ValueError(f"Missing command/arguments for {entry['file']}")
        arguments = shlex.split(command)
    if (
        not isinstance(arguments, list)
        or not arguments
        or not all(isinstance(a, str) for a in arguments)
        or not arguments[0]
    ):
        raise ValueError(f"Expected nonempty string arguments for {entry['file']}")
    if any(a in ("&&", "||", ";", "|") for a in arguments):
        raise ValueError(
            f"Expected one compiler command, not a shell pipeline: {entry['file']}"
        )

    prefix = (
        prefix.resolve()
        if prefix is not None
        else environment_prefix(arguments, Path(entry["directory"]))
    )
    extra = []
    if prefix is not None:
        flags = (
            ["-isystem", str(prefix / "include")]
            if has_option(arguments, "--gcc-install-dir", "--gcc-toolchain")
            else toolchain_flags(prefix)
        )
        # Recognize the common joined and separate forms without rewriting them.
        include = str(prefix / "include")
        present = any(
            (
                a in ("-I", "-isystem", "--system-include")
                and i + 1 < len(arguments)
                and arguments[i + 1] == include
            )
            or a
            in (
                f"-I{include}",
                f"-isystem{include}",
                f"-isystem={include}",
                f"--system-include={include}",
            )
            for i, a in enumerate(arguments)
        )
        if not present:
            extra.extend(flags[:2])
        aliases = {
            "--target": ("-target",),
            "--sysroot": ("-isysroot",),
            "--gcc-install-dir": ("--gcc-toolchain",),
        }
        for flag in flags[2:]:
            name = flag.split("=", 1)[0]
            if not has_option(arguments, name, *aliases.get(name, ())):
                extra.append(flag)

    if Path(arguments[0]).name == "nvcc":
        if prefix is not None and not has_option(arguments, "--cuda-path"):
            # CUDA packages use either the traditional or the split layout.
            roots = (
                [prefix]
                if (prefix / "include/cuda.h").is_file()
                else sorted(
                    p.parent.parent
                    for p in (prefix / "targets").glob("*/include/cuda.h")
                )
            )
            if len(roots) != 1:
                raise ValueError(
                    f"Expected one CUDA toolkit in {prefix}; add --cuda-path to the command"
                )
            extra.append(f"--cuda-path={roots[0]}")
        if not has_option(arguments, "--cuda-gpu-arch", "--offload-arch"):
            arch = cuda_gpu_arch or cuda_architecture(arguments)
            if arch:
                extra.append(f"--cuda-gpu-arch={arch}")

    converted = {k: v for k, v in entry.items() if k != "command"}
    converted["arguments"] = [arguments[0], *extra, *arguments[1:]]
    return converted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--prefix",
        type=Path,
        help="environment (default: compiler location, then CONDA_PREFIX)",
    )
    parser.add_argument(
        "--cuda-gpu-arch",
        help="architecture, e.g. sm_89 (default: first NVCC architecture)",
    )
    options = parser.parse_args(argv)
    temporary = None
    try:
        source = options.source.resolve()
        destination = options.destination.resolve()
        if source == destination or (
            destination.exists() and source.samefile(destination)
        ):
            raise ValueError("Source and destination must be different files")
        entries = json.loads(source.read_text())
        if not isinstance(entries, list):
            raise ValueError("Compilation database must be a JSON array")
        converted = [
            convert_entry(
                e,
                prefix=options.prefix,
                cuda_gpu_arch=options.cuda_gpu_arch,
            )
            for e in entries
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Validate everything first; an interrupted write cannot truncate a CDB.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(converted, stream, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {len(converted)} entries to {destination}")
