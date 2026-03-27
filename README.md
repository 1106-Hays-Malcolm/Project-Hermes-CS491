# Project-Hermes-CS491

Our repository for our CS 491 final project.

## Git Submodues

To add your own repository as a submodule, you must do the following command...

```bash
git submodule add <repository URL> [path]
# of note, path must refer to folder/path that doesn't exist yet (it'll create it)
```

```bash
# example for RAG subomdule
git submodule add https://github.com/PieFlavr/CS491-RAG RAG
```

### Submodule Updating

On the initial clone of the project, to update submodules to latest version run the following.

```bash
git submodule update --init --recursive
```

To update the submodule version in the repository, run the following...

```bash
git subomdule update --remote <submodule path>
```

```bash
# example for RAG submodule
git submodule update --remote RAG
```
