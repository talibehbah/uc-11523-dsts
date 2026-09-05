# JupyterLab on Windows: SSL certificate error fix

If `jupyter lab` fails to start in your conda environment with an SSL / ASN1 error, this guide is for you. This is a **known Windows + Python issue**, not a missing JupyterLab package.

---

## Symptoms

After activating your env (for example `fusionapp`) and running:

```powershell
jupyter lab
```

you see a traceback that ends something like:

```text
File "...\tornado\netutil.py", line 35, in <module>
    _client_ssl_defaults = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
...
File "...\ssl.py", line ..., in _load_windows_store_certs
    self.load_verify_locations(cadata=certs)
ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4057)
```

JupyterLab may already be installed; the crash happens while Python is setting up SSL during import of Tornado / Jupyter Server.

---

## What causes it (plain language)

On Windows, Python often loads trusted certificates from the **Windows certificate store** when creating a default SSL context.

If that store contains a **corrupt or incomplete certificate**, Python can raise `ASN1: NOT_ENOUGH_DATA` and crash **before** JupyterLab finishes starting.

Important:

- This is **not** usually fixed by reinstalling JupyterLab alone.
- Setting `SSL_CERT_FILE` to a certifi bundle is helpful for HTTPS trust, but **by itself it is not enough** for this ASN1 crash, because Python still tries to load the Windows store.

---

## Quick fix (recommended)

Do this inside the conda environment where JupyterLab fails (examples below use PowerShell; Anaconda Prompt is fine too).

### 1. Activate your environment

```powershell
conda activate fusionapp
```

Use your real env name if it is different.

### 2. Install / confirm `certifi`

```powershell
python -c "import certifi; print(certifi.where())"
```

If that fails:

```powershell
conda install -c conda-forge certifi
```

or:

```powershell
pip install certifi
```

### 3. Add a small startup patch (`sitecustomize.py`)

Python automatically runs `sitecustomize.py` if it exists in your env’s `site-packages` folder.

1. Find your env prefix (while the env is active):

```powershell
echo $env:CONDA_PREFIX
```

2. Create / edit this file:

```text
%CONDA_PREFIX%\Lib\site-packages\sitecustomize.py
```

Example full path shape:

```text
C:\Users\<You>\.conda\envs\fusionapp\Lib\site-packages\sitecustomize.py
```

3. Paste the following contents (save as UTF-8):

```python
"""Windows SSL workaround for corrupt/broken certs in the Windows certificate store.

Python's ssl.create_default_context() calls SSLContext._load_windows_store_certs(),
which can raise ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] when a bad cert is present.
That breaks tornado/jupyter_server import and prevents `jupyter lab` from starting.

This skips unloadable store certs instead of crashing, and prefers certifi's CA bundle
when SSL_CERT_FILE / REQUESTS_CA_BUNDLE are not already set.

You can remove this file after upgrading Python (if the upstream fix is present)
or after cleaning bad Windows certificates.
"""
import os
import ssl
import warnings

try:
    import certifi
    _ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", _ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
except Exception:
    pass

_orig = getattr(ssl.SSLContext, "_load_windows_store_certs", None)
if _orig is not None and not getattr(ssl.SSLContext, "_windows_cert_store_patch", False):
    def _load_windows_store_certs_safe(self, storename, purpose):
        try:
            return _orig(self, storename, purpose)
        except ssl.SSLError as e:
            warnings.warn(
                f"Skipping broken certificates from Windows store {storename!r}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return bytearray()

    ssl.SSLContext._load_windows_store_certs = _load_windows_store_certs_safe
    ssl.SSLContext._windows_cert_store_patch = True
```

### 4. Make certifi the default CA bundle for this env (permanent)

Still with the env activated:

```powershell
python -c "import certifi; print(certifi.where())"
```

Copy the printed path, then set conda env vars (replace the path with yours):

```powershell
conda env config vars set SSL_CERT_FILE="C:\Users\<You>\.conda\envs\fusionapp\Lib\site-packages\certifi\cacert.pem"
conda env config vars set REQUESTS_CA_BUNDLE="C:\Users\<You>\.conda\envs\fusionapp\Lib\site-packages\certifi\cacert.pem"
```

Or in one go (PowerShell):

```powershell
$ca = python -c "import certifi; print(certifi.where())"
conda env config vars set "SSL_CERT_FILE=$ca" "REQUESTS_CA_BUNDLE=$ca"
```

**Re-activate** the environment so the new variables load:

```powershell
conda deactivate
conda activate fusionapp
```

Check:

```powershell
conda env config vars list
```

---

## Session-only reminder (optional)

For the current PowerShell window only:

```powershell
conda activate fusionapp
$env:SSL_CERT_FILE = (python -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
```

Remember: without the `sitecustomize.py` patch, this usually **will not** stop the ASN1 crash.

---

## How to verify

```powershell
conda activate fusionapp
jupyter lab --version
jupyter lab
```

If `--version` prints a version (for example `4.6.3`) and Lab opens in the browser, the fix is working.

You may still see **UserWarnings** mentioning `NOT_ENOUGH_DATA` / skipping Windows store certs. Those warnings are expected with this workaround and are usually **harmless**.

---

## Notes and safety

- **SSL verification is still enabled.** The patch does not turn SSL off; it skips certificates that Python cannot parse, and prefers the certifi CA bundle for trust roots.
- **This is an environment-local fix** (`sitecustomize.py` and conda env vars affect that conda env, not every Python on the machine).
- **Optional longer-term cleanup:** open Windows Certificate Manager (`Win + R` → `certmgr.msc`) and review Trusted Root / Intermediate Certification Authorities for odd or corrupted entries. Only remove certificates you understand; ask IT/tutor if unsure.
- To undo the workaround later:

```powershell
Remove-Item "$env:CONDA_PREFIX\Lib\site-packages\sitecustomize.py"
conda env config vars unset SSL_CERT_FILE REQUESTS_CA_BUNDLE
conda deactivate
conda activate fusionapp
```

---

## Quick checklist

1. `conda activate <your-env>`
2. Confirm `certifi` works
3. Save `sitecustomize.py` under `%CONDA_PREFIX%\Lib\site-packages\`
4. Set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` with `conda env config vars set`
5. Re-activate the env
6. Run `jupyter lab --version`, then `jupyter lab`

If you are stuck after these steps, tell your tutor: your conda env name, the exact traceback tail, and whether `sitecustomize.py` exists under `%CONDA_PREFIX%\Lib\site-packages\`.
