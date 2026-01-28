# SAM3 Model Download

Use one of the methods below to download SAM3 weights without access authorization.

## Method A: ModelScope CLI (recommended)

Prerequisite: Python 3.8+ with `pip` available.

```
# install modelscope
pip install modelscope

# download SAM3 model
modelscope download --model facebook/sam3
```

The files will be placed in your ModelScope cache directory. You can check it with:

```
modelscope config --get cache_dir
```

## Method B: Using a pre-downloaded archive

If you already have a SAM3 weight archive, place the files into this folder:

```
./models
```

Make sure the file names and directory layout match what your inference script expects.



