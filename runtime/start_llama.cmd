@echo off
REM Start the local llama-server. Official llama.cpp build - auto-selects the
REM alderlake CPU backend on this machine (~19 tok/s vs 0.5 for the pip wheel).
setlocal
set MODEL=%~1
if "%MODEL%"=="" for %%f in ("%~dp0..\quantization\builds\vjlm*.gguf") do set MODEL=%%f
if "%MODEL%"=="" set MODEL=%~dp0..\quantization\builds\base-qwen2.5-1.5b-q4_k_m.gguf
echo Serving %MODEL%
"%~dp0llama\llama-server.exe" -m "%MODEL%" -c 4096 -t 8 --host 127.0.0.1 --port 8080
