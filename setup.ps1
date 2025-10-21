Param()

$ErrorActionPreference = "Stop"

$envName = "fl_privacy"
$pythonVersion = "3.9"

function Assert-CondaInstalled {
    try {
        conda --version | Out-Null
    } catch {
        Write-Error "Conda not found. Please install Miniconda/Anaconda and ensure 'conda' is available in PowerShell (run 'conda init powershell')."
    }
}

function Env-Exists($name) {
    $list = conda env list | Out-String
    return $list -match "(?m)^$name\s"
}

Write-Host "Setting up environment '$envName' for Windows..." -ForegroundColor Cyan

Assert-CondaInstalled

if (Env-Exists $envName) {
    Write-Host "Removing existing environment: $envName" -ForegroundColor Yellow
    conda env remove -n $envName -y | Out-Null
}

Write-Host "Creating conda environment: $envName (Python $pythonVersion)" -ForegroundColor Cyan
conda create -n $envName python=$pythonVersion -y | Out-Null

Write-Host "Upgrading pip in $envName" -ForegroundColor Cyan
conda run -n $envName python -m pip install --upgrade pip | Out-Null

Write-Host "Installing PyTorch (CPU/CUDA as per pip defaults)" -ForegroundColor Cyan
conda run -n $envName python -m pip install torch torchvision | Out-Null

$reqPath = Join-Path $PSScriptRoot 'requirements.txt'
if (Test-Path $reqPath) {
    Write-Host "Installing additional requirements from requirements.txt" -ForegroundColor Cyan
    conda run -n $envName python -m pip install -r $reqPath | Out-Null
} else {
    Write-Warning "requirements.txt not found. Installing a minimal set of packages..."
    conda run -n $envName python -m pip install numpy pandas matplotlib seaborn scikit-image hydra-core pyyaml tenseal | Out-Null
}

Write-Host "" 
Write-Host "Setup Complete -------------------" -ForegroundColor Green
Write-Host "To activate this environment in a new shell, run:" -NoNewline; Write-Host "  conda activate $envName" -ForegroundColor Yellow
Write-Host "If activation fails in PowerShell, run 'conda init powershell' once, restart the shell, and try again." -ForegroundColor DarkGray

