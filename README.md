# GraphActiveSearch

# Directory Tree

# Docker Workflow

This project was developed on **macOS** (M3 pro chip). The code is guaranteed to run using Docker on **macOS** and **Windows**. **Linux** functionality has not be tested. 

Two scripts are available (see also **Directory Tree**):

- `src/active_search_obsTopo.jl`
- `src/active_search_full.jl`

The default script is `active_search_obsTopo.jl`.  
Users can override this at runtime to execute the other version.

---


## 1️⃣ macOS Machines

Users can either **load the prebuilt image** (see below for donwload link) or **rebuild locally**.

---

### Option A — Load Prebuilt Image

#### Unzip

```bash
gunzip graph-active-search-mac.tar.gz
```

#### Load

```bash
docker load -i graph-active-search-mac.tar
```

#### Run Default Script

```bash
docker run --rm -v $(pwd):/active_search graph-active-search:mac
```

#### Run Other Script

```bash
docker run --rm -v $(pwd):/active_search graph-active-search:mac \
    julia --project=. src/active_search_full.jl
```

---

### Option B — Rebuild Instead

#### Apple Silicon

```bash
docker build --platform linux/arm64 -t graph-active-search:mac .
```

#### Intel

```bash
docker build --platform linux/amd64 -t graph-active-search:mac .
```

Then run using the same commands as above.

---

### Open Output

```bash
open output\<name>.html
```

---

## 2️⃣ Windows (Rebuild Locally)

Windows users should rebuild the image. Loading the prebuilt image is not guaranteed to work.

---

### Install Docker Desktop

- Enable **WSL2 backend**

---

### Open PowerShell in Project Folder

Example:

```
C:\Users\X\Desktop\GraphActiveSearch
```

---

### Build (amd64)

```powershell
docker build --platform linux/amd64 -t graph-active-search:win .
```

---

### Run Default Script

#### PowerShell

```powershell
docker run --rm -v ${PWD}.Path:/active_search graph-active-search:win
```

#### CMD

```cmd
docker run --rm -v %cd%:/active_search graph-active-search:win
```

---

### Run Other Script

#### PowerShell

```powershell
docker run --rm -v ${PWD}.Path:/active_search graph-active-search:win `
    julia --project=. src/active_search_full.jl
```

#### CMD

```cmd
docker run --rm -v %cd%:/active_search graph-active-search:win julia --project=. src/active_search_full.jl
```

---

### Open Output

```powershell
start output\<name>.html
```

---

## Output

Generated files appear in:

```
output/
```

Example:

```
output/2026-03-04_164506_performance.png
output/2026-03-04_164506_tree.html
```

---

## Notes

- No multi-architecture images are used.
- macOS images are not intended for Windows.
- Windows should rebuild locally.
- All output paths are relative and OS-independent.

# Download the Docker Image (! Large File)

The Docker image for this project can be downloaded directly:

[Download graph-active-search.tar](https://drive.google.com/uc?export=download&id=1tKyE9RUXGm_TrY5ZblIiGt4ai4Zp2o7y)

> **Note:** This is a large file (~620 MB). Make sure you have enough space and a stable internet connection.
