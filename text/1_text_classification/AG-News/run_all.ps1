# Train all three cells, then score each on the held-out test split.
#
# Usage (from the AG-News project root, with the torch env active):
#     .\run_all.ps1
#     .\run_all.ps1 -Cells rnn,lstm          # subset
#     .\run_all.ps1 -TrainArgs '--seed',7    # extra flags passed to train.py
#
# If PowerShell refuses to run the file ("running scripts is disabled"):
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#
# Stops at the first failure -- a crashed run leaves a half-written log, and
# continuing would bury the error under thousands of lines of the next run.

param(
    [string[]]$Cells = @('rnn', 'gru', 'lstm'),
    [string[]]$TrainArgs = @()
)

$ErrorActionPreference = 'Stop'
$start = Get-Date

foreach ($cell in $Cells) {
    Write-Host "`n===== train $cell =====" -ForegroundColor Cyan
    python train.py --cell $cell @TrainArgs
    # $LASTEXITCODE is how a native exe reports failure; $? is not reliable here.
    if ($LASTEXITCODE -ne 0) { throw "train.py --cell $cell exited with $LASTEXITCODE" }
}

foreach ($cell in $Cells) {
    Write-Host "`n===== test $cell =====" -ForegroundColor Cyan
    python eval.py --cell $cell --split test --save-cm
    if ($LASTEXITCODE -ne 0) { throw "eval.py --cell $cell exited with $LASTEXITCODE" }
}

$mins = ((Get-Date) - $start).TotalMinutes
Write-Host ("`nAll done in {0:N1} min. Results: outputs_<cell>/training_log.json" -f $mins) -ForegroundColor Green
