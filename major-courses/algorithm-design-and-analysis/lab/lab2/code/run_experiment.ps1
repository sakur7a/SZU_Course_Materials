Param(
    [string]$Compiler = "g++",
    [string]$Python = "python",
    [int]$Repeats = 3,
    [int]$BruteLimit = 20000,
    [int]$ValidateN = 3000,
    [int]$DemoN = 150,
    [int]$BenchmarkStart = 100000,
    [int]$BenchmarkEnd = 1000000,
    [int]$BenchmarkStep = 100000,
    [int]$Seed = 20260416,
    [switch]$Quick,
    [switch]$NoGuiWindow,
    [switch]$SkipBenchmark,
    [switch]$SkipVisualization,
    [switch]$SkipReport,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"

if ($Quick) {
    if (-not $PSBoundParameters.ContainsKey("Repeats")) { $Repeats = 1 }
    if (-not $PSBoundParameters.ContainsKey("BenchmarkStart")) { $BenchmarkStart = 20000 }
    if (-not $PSBoundParameters.ContainsKey("BenchmarkEnd")) { $BenchmarkEnd = 200000 }
    if (-not $PSBoundParameters.ContainsKey("BenchmarkStep")) { $BenchmarkStep = 20000 }
    if (-not $PSBoundParameters.ContainsKey("ValidateN")) { $ValidateN = 1200 }
    if (-not $PSBoundParameters.ContainsKey("DemoN")) { $DemoN = 120 }
}

Push-Location $PSScriptRoot
try {
    $resultsDir = Resolve-Path ".."
    $resultsDir = Join-Path $resultsDir "results"
    if (-not (Test-Path $resultsDir)) {
        New-Item -ItemType Directory -Path $resultsDir | Out-Null
    }

    if (-not (Get-Command $Compiler -ErrorAction SilentlyContinue)) {
        throw "Compiler '$Compiler' not found in PATH."
    }
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
        throw "Python interpreter '$Python' not found in PATH."
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Configuration: repeats=$Repeats brute_limit=$BruteLimit validate_n=$ValidateN demo_n=$DemoN" -ForegroundColor Cyan
    Write-Host "Benchmark range: $BenchmarkStart..$BenchmarkEnd step=$BenchmarkStep quick=$Quick" -ForegroundColor Cyan

    Write-Host "[1/6] Compiling C++ program..." -ForegroundColor Yellow
    & $Compiler "./closest_pair.cpp" "-O2" "-std=c++17" "-o" "./closest_pair.exe"

    Write-Host "[2/6] Validating brute force vs divide and conquer..." -ForegroundColor Yellow
    & "./closest_pair.exe" --mode validate --n $ValidateN --seed $Seed

    if (-not $SkipExport) {
        Write-Host "[3/6] Exporting demo point set with divide-and-conquer trace..." -ForegroundColor Yellow
        & "./closest_pair.exe" --mode export-divcon --n $DemoN --seed $Seed --points-csv "../results/points_demo.csv" --pair-csv "../results/pair_demo.csv" --trace-csv "../results/trace_demo.csv" --steps-csv "../results/steps_demo.csv"
    }

    if (-not $SkipBenchmark) {
        Write-Host "[4/6] Running benchmark..." -ForegroundColor Yellow
        & "./closest_pair.exe" --mode benchmark --n-start $BenchmarkStart --n-end $BenchmarkEnd --step $BenchmarkStep --repeats $Repeats --brute-limit $BruteLimit --seed $Seed --output "../results/benchmark.csv"
    }

    if ((-not $SkipVisualization) -and (-not $SkipBenchmark)) {
        Write-Host "[5/6] Generating benchmark charts..." -ForegroundColor Yellow
        & $Python "../viz/plot_benchmark.py" --input "../results/benchmark.csv" --output-prefix "../results"
    }

    if ((-not $SkipVisualization) -and (-not $SkipExport)) {
        Write-Host "[6/6] Launching process visualization window..." -ForegroundColor Yellow
        $vizArgs = @(
            "../viz/visualize_process.py",
            "--points", "../results/points_demo.csv",
            "--pair", "../results/pair_demo.csv",
            "--trace", "../results/trace_demo.csv",
            "--steps", "../results/steps_demo.csv",
            "--save", "../results/process_dashboard.png",
            "--save-gif", "../results/process_dashboard.gif",
            "--view", "dashboard",
            "--no-show"
        )
        & $Python @vizArgs
    }

    if ((-not $SkipReport) -and (-not $SkipBenchmark)) {
        Write-Host "[Final] Generating markdown report..."
        & $Python "../report/generate_report.py" --benchmark "../results/benchmark.csv" --output "../report/report_closest_pair.md"
    }

    $sw.Stop()
    Write-Host ("Done in {0:n2}s. Please check ../results and ../report folders." -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}
finally {
    Pop-Location
}
