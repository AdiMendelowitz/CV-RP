# PowerShell Setup Script for ml-research-12weeks
# Windows 11 - PyCharm - Python 3.13 - venv
# Run: .\setup.ps1

Write-Host "Creating 12-week ML Research project structure..." -ForegroundColor Green
Write-Host ""

# Create week folders with subfolders
Write-Host "Creating week folders (12 total)..." -ForegroundColor Cyan
for ($i=1; $i -le 12; $i++) {
    $weekNum = "{0:D2}" -f $i
    $weekFolder = "week-$weekNum"
    
    New-Item -ItemType Directory -Path $weekFolder -Force | Out-Null
    New-Item -ItemType Directory -Path "$weekFolder\code" -Force | Out-Null
    New-Item -ItemType Directory -Path "$weekFolder\notebooks" -Force | Out-Null
    New-Item -ItemType Directory -Path "$weekFolder\experiments" -Force | Out-Null
    New-Item -ItemType Directory -Path "$weekFolder\notes" -Force | Out-Null
    
    Write-Host "  Created $weekFolder" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Creating support folders..." -ForegroundColor Cyan

# Resources
New-Item -ItemType Directory -Path "resources" -Force | Out-Null
New-Item -ItemType Directory -Path "resources\datasets" -Force | Out-Null
New-Item -ItemType Directory -Path "resources\models" -Force | Out-Null
New-Item -ItemType Directory -Path "resources\configs" -Force | Out-Null
Write-Host "  Created resources/" -ForegroundColor Gray

# Papers
New-Item -ItemType Directory -Path "papers" -Force | Out-Null
Write-Host "  Created papers/" -ForegroundColor Gray

# Blog drafts
New-Item -ItemType Directory -Path "blog-drafts" -Force | Out-Null
Write-Host "  Created blog-drafts/" -ForegroundColor Gray

# Portfolio
New-Item -ItemType Directory -Path "portfolio" -Force | Out-Null
New-Item -ItemType Directory -Path "portfolio\images" -Force | Out-Null
New-Item -ItemType Directory -Path "portfolio\demos" -Force | Out-Null
New-Item -ItemType Directory -Path "portfolio\docs" -Force | Out-Null
Write-Host "  Created portfolio/" -ForegroundColor Gray

Write-Host ""
Write-Host "Creating .gitignore..." -ForegroundColor Cyan

# Create .gitignore content
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# PyCharm
.idea/
*.iml

# Virtual Environment
venv/
.venv/
ENV/
env/

# Jupyter
.ipynb_checkpoints/

# Data files
*.h5
*.hdf5
*.pkl
*.pickle
*.pt
*.pth
*.ckpt
*.weights

# Datasets and Models
resources/datasets/*
!resources/datasets/.gitkeep
resources/models/*
!resources/models/.gitkeep

# Papers PDFs
papers/**/*.pdf

# Logs
*.log
wandb/
mlruns/
tensorboard_logs/

# Windows
Thumbs.db
desktop.ini

# Temporary
*.swp
*~
.tmp/
"@

# Write .gitignore
$gitignore | Out-File -FilePath ".gitignore" -Encoding utf8

Write-Host "  Created .gitignore" -ForegroundColor Gray

# Create .gitkeep files
Write-Host ""
Write-Host "Creating .gitkeep files..." -ForegroundColor Cyan
New-Item -ItemType File -Path "resources\datasets\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "resources\models\.gitkeep" -Force | Out-Null
Write-Host "  Created .gitkeep files" -ForegroundColor Gray

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "SUCCESS! Project structure created!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Project structure:" -ForegroundColor Yellow
Write-Host "  - week-01 through week-12 (with code, notebooks, experiments, notes)" -ForegroundColor White
Write-Host "  - resources (datasets, models, configs)" -ForegroundColor White
Write-Host "  - papers" -ForegroundColor White
Write-Host "  - blog-drafts" -ForegroundColor White
Write-Host "  - portfolio (images, demos, docs)" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify folders in PyCharm Project panel" -ForegroundColor White
Write-Host "  2. Install Python packages (see SETUP_GUIDE.md)" -ForegroundColor White
Write-Host "  3. Open week-01\README.md to see tasks" -ForegroundColor White
Write-Host "  4. Start coding!" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding!" -ForegroundColor Magenta
