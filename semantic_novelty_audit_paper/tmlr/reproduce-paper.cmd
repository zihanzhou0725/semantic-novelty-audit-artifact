@echo off
rem Wrapper for systems whose PowerShell execution policy is Restricted.
rem Double-click this file, or run: reproduce-paper.cmd [ps-args...]
rem All arguments are forwarded to reproduce-paper.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reproduce-paper.ps1" %*
