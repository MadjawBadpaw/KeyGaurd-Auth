@echo off
pip install pyinstaller bcrypt
pyinstaller --onefile --noconsole --name KeyGuard ^
  --add-data "static;static" ^
  --hidden-import sklearn.ensemble._iforest ^
  --hidden-import sklearn.utils._cython_blas ^
  --hidden-import sklearn.neighbors._typedefs ^
  --hidden-import sklearn.tree._utils ^
  --hidden-import pynput.keyboard._win32 ^
  --hidden-import pynput.mouse._win32 ^
  --hidden-import uvicorn.logging ^
  --hidden-import uvicorn.loops.auto ^
  --hidden-import uvicorn.protocols.http.auto ^
  --hidden-import uvicorn.lifespan.off ^
  --hidden-import bcrypt ^
  run.py
echo.
echo Done — dist\KeyGuard.exe
pause