Handle the chores of compile commands.

```sh
make-clangd-db SOURCE DESTINATION
```

Both paths are required and must refer to different files.

The environment is detected from the compiler path or `CONDA_PREFIX`. Override it with `--prefix PATH`. See `prepare-clangd-db --help` for all options.
