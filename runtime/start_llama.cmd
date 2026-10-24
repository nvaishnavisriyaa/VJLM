@echo off
REM Start the local llama-server for VJLM. CPU only - that is the deployment
REM requirement, not a limitation to work around.
REM
REM Updated 2026-09-19 for Phase 9 (QUANT-001). What changed and why:
REM
REM   * Binaries moved to runtime\llama\cpu\. The pinned build is recorded in
REM     runtime\llama\PINNED_TAG.txt and must match the build every number in
REM     experiments\quant_results.jsonl came from.
REM   * -t 8 removed. That came from a 14-core laptop this project no longer
REM     runs on; this machine has 20 physical cores. Thread count is set from
REM     measurement, because more threads is NOT monotonically faster - past
REM     the memory-bandwidth limit extra threads just contend.
REM   * The Qwen-1.5B fallback path is gone. It pointed at a file that has
REM     never existed on this machine, so the "fallback" only ever turned a
REM     clear error into a confusing one.
REM
setlocal
set LLAMA=%~dp0llama\llama-server.exe
if not exist "%LLAMA%" (
  echo ERROR: %LLAMA% not found.
  echo Download the pinned llama.cpp build - see runtime\llama\PINNED_TAG.txt
  exit /b 1
)

REM Threads: set from the sweep in QUANT-001, not guessed. Override with
REM   set VJLM_THREADS=n
if "%VJLM_THREADS%"=="" set VJLM_THREADS=20

REM Model: explicit argument wins. Otherwise the build selected by QUANT-001.
set MODEL=%~1
if "%VJLM_MODEL%"=="" set VJLM_MODEL=vjlm-llama3b-Q6_K-imat.gguf
if "%MODEL%"=="" set MODEL=%~dp0..\quantization\builds\%VJLM_MODEL%
if "%MODEL%"=="" goto :nomodel
if not exist "%MODEL%" (
  echo ERROR: model not found: %MODEL%
  echo Pass a .gguf path explicitly, or set VJLM_MODEL to a file in
  echo   quantization\builds\gguf\
  echo No fallback is attempted on purpose - serving a different model than
  echo the one that was benchmarked would invalidate every recorded number.
  exit /b 1
)

echo Serving %MODEL%
echo Threads %VJLM_THREADS%  ^|  context 4096  ^|  CPU only
"%LLAMA%" -m "%MODEL%" -c 4096 -t %VJLM_THREADS% --host 127.0.0.1 --port 8080
goto :eof

:nomodel
echo ERROR: no model specified.
echo   usage: start_llama.cmd path\to\model.gguf
echo   or:    set VJLM_MODEL=vjlm-llama3b-Q6_K-imat.gguf ^&^& start_llama.cmd
exit /b 1
