# Complete Setup Guide - Windows 11 + PyCharm + venv

**Your Configuration:**
- OS: Windows 11 Lenovo
- IDE: PyCharm
- Python: 3.12
- Environment: venv (virtual environment)
- Path: `C:\Users\adime\Documents\Career\ml-research-12weeks`

---

## 📋 Table of Contents

1. [Fix Invalid Python Interpreter](#fix-invalid-interpreter)
2. [Create Folder Structure](#create-folder-structure)
3. [Install Python Packages](#install-packages)
4. [Manual Folder Creation](#manual-folder-creation)
5. [Troubleshooting](#troubleshooting)

---

## 🔧 Fix Invalid Interpreter

**Problem:** PyCharm shows "[invalid] Python 3.12 (ml-research-12weeks)" in red.

**Solution:**

### **Step 1: Remove Invalid Interpreters**

1. **Open Settings:** `Ctrl + Alt + S`

2. **Navigate to:** Project: ml-research-12weeks → Python Interpreter

3. **Click gear icon** ⚙️ → **"Show All..."**

4. **Delete invalid interpreters:**
   - You'll see a list of interpreters
   - Find any with **"invalid"** or **red text**
   - Select each → Click **minus (-)** button
   - Delete ALL invalid ones

5. **Click OK**

### **Step 2: Create Fresh Interpreter**

1. **Still in Settings → Python Interpreter**

2. **Click gear icon** ⚙️ → **"Add..."** or **"Add Interpreter..."**

3. **Select "Virtualenv Environment"** (left sidebar)

4. **Choose "New"** radio button

5. **Configure settings:**

   **Location:** Should auto-fill to:
   ```
   C:\Users\adime\Documents\Career\ml-research-12weeks\.venv
   ```

   **Base interpreter:** Click **"..."** button and select:
   ```
   C:\Users\adime\AppData\Local\Programs\Python\Python313\python.exe
   ```
   
   **If you can't find it:** In PowerShell run:
   ```powershell
   where.exe python
   # Or
   python -c "import sys; print(sys.executable)"
   ```

6. **Uncheck these options:**
   - ☐ Inherit global site-packages
   - ☐ Make available to all projects

7. **Click OK**

8. **Wait 30-60 seconds** for PyCharm to create venv

### **Step 3: Verify It Works**

**In PyCharm Terminal (bottom panel):**

```powershell
# Should see (.venv) at the start of prompt
(.venv) PS C:\Users\adime\Documents\Career\ml-research-12weeks>

# Check Python version
python --version
# Output: Python 3.12.x

# Check pip works
pip --version
# Output: pip X.X.X from ...\.venv\...

# Test import
python -c "print('Python works!')"
# Output: Python works!
```

**If all three work:** ✅ You're good!

---

## 📁 Create Folder Structure

### **Option A: Use PowerShell Script** ⭐ **RECOMMENDED**

**In PyCharm Terminal:**

```powershell
# Run the script
.\setup.ps1
```

**If you get execution policy error:**

```powershell
# Bypass policy for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Then run script
.\setup.ps1
```

**What it creates:**
- week-01 through week-12 (each with code, notebooks, experiments, notes subfolders)
- resources (datasets, models, configs)
- papers
- blog-drafts
- portfolio (images, demos, docs)
- .gitignore file
- .gitkeep files

**Time:** ~10 seconds

---

### **Option B: Manual Creation** 

**If script doesn't work, create manually in PyCharm:**

1. **Right-click project root** (ml-research-12weeks) → New → Directory

2. **Create these folders:**

**Week folders (12 total):**
```
week-01
  ├── code
  ├── notebooks
  ├── experiments
  └── notes

week-02
  ├── code
  ├── notebooks
  ├── experiments
  └── notes

... (repeat for week-03 through week-12)
```

**Support folders:**
```
resources
  ├── datasets
  ├── models
  └── configs

papers

blog-drafts

portfolio
  ├── images
  ├── demos
  └── docs
```

**Time:** ~3-5 minutes

---

## 📦 Install Packages

**Make sure you're in venv first!** Look for `(.venv)` in terminal.

**In PyCharm Terminal, copy-paste these commands:**

### **Step 1: PyTorch (3-4 minutes)**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Wait for this to complete before continuing.**

### **Step 2: Core ML Packages (2 minutes)**

```powershell
pip install numpy scipy pandas scikit-learn matplotlib seaborn
```

### **Step 3: Computer Vision (2 minutes)**

```powershell
pip install opencv-python albumentations timm ultralytics
```

### **Step 4: Time Series (2 minutes)**

```powershell
pip install statsmodels prophet neuralprophet
```

**Note:** If prophet fails, try:
```powershell
pip install prophet --no-cache-dir
```

### **Step 5: Development Tools (2 minutes)**

```powershell
pip install wandb jupyterlab black pytest ipykernel
```

### **Step 6: Utilities (1 minute)**

```powershell
pip install gradio streamlit tqdm pillow
```

### **Step 7: Save Environment**

```powershell
pip freeze > requirements.txt
```

**Total time:** ~10-12 minutes

### **Verify Installation**

```powershell
# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# Test OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test other packages
python -c "import numpy, pandas, matplotlib; print('Core packages OK')"
```

**If all print successfully:** ✅ All packages installed!

---

## 🛠️ Manual Folder Creation

**Detailed steps if you prefer doing it manually:**

### **In PyCharm Project Panel (left side):**

**1. Create Week Folders:**

For each week (01 through 12):
- Right-click project → New → Directory → `week-01`
- Right-click `week-01` → New → Directory → `code`
- Right-click `week-01` → New → Directory → `notebooks`
- Right-click `week-01` → New → Directory → `experiments`
- Right-click `week-01` → New → Directory → `notes`
- Repeat for week-02 through week-12

**2. Create Resources Folder:**

- Right-click project → New → Directory → `resources`
- Right-click `resources` → New → Directory → `datasets`
- Right-click `resources` → New → Directory → `models`
- Right-click `resources` → New → Directory → `configs`

**3. Create Papers Folder:**

- Right-click project → New → Directory → `papers`

**4. Create Blog Drafts Folder:**

- Right-click project → New → Directory → `blog-drafts`

**5. Create Portfolio Folder:**

- Right-click project → New → Directory → `portfolio`
- Right-click `portfolio` → New → Directory → `images`
- Right-click `portfolio` → New → Directory → `demos`
- Right-click `portfolio` → New → Directory → `docs`

**Total time:** 3-5 minutes

---

## 🆘 Troubleshooting

### **"Could not find platform independent libraries"**

**Problem:** Spaces or special characters in path.

**Solution:**
- Your current path is fine: `C:\Users\adime\Documents\Career\ml-research-12weeks`
- No spaces after "Career" ✅
- If you still get this error, try:
  ```powershell
  python -m pip install package_name
  ```
  Instead of:
  ```powershell
  pip install package_name
  ```

---

### **"pip not recognized"**

**Problem:** venv not activated or pip not in PATH.

**Solution:**

**Check if venv is active:**
```powershell
# Should see (.venv) at start
(.venv) PS C:\Users\adime\...
```

**If not active, close and reopen PyCharm.** PyCharm should auto-activate.

**Alternative:**
```powershell
# Use Python module
python -m pip install package_name
```

---

### **"PowerShell script won't run"**

**Problem:** Execution policy restriction.

**Solution:**

**Temporary bypass (safest):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

**Or run commands manually** (see Manual Folder Creation section).

---

### **"Can't find Python 3.12"**

**Problem:** Python not in expected location.

**Solution:**

**Find your Python:**
```powershell
where.exe python
```

**Or:**
```powershell
python -c "import sys; print(sys.executable)"
```

**Common locations:**
- `C:\Users\adime\AppData\Local\Programs\Python\Python313\python.exe`
- `C:\Python313\python.exe`
- `C:\Users\adime\anaconda3\python.exe`

**Use the path shown in PyCharm when adding interpreter.**

---

### **"Package installation fails"**

**Problem:** Various causes.

**Solutions:**

**Try these in order:**

1. **Update pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

2. **Clear pip cache:**
   ```powershell
   pip cache purge
   ```

3. **Install without cache:**
   ```powershell
   pip install package_name --no-cache-dir
   ```

4. **Use Python module:**
   ```powershell
   python -m pip install package_name
   ```

5. **For prophet specifically:**
   ```powershell
   # Install dependencies first
   pip install numpy cython
   pip install prophet
   ```

---

### **"venv not activating"**

**Problem:** PyCharm not recognizing venv.

**Solution:**

**Method 1: Restart PyCharm**
- Close PyCharm completely
- Reopen project
- Check for (.venv) in terminal

**Method 2: Reconfigure interpreter**
- Settings → Python Interpreter
- Change to different interpreter
- Change back to .venv interpreter
- OK

**Method 3: Recreate venv**
- Delete `.venv` folder
- Settings → Python Interpreter → Add → New Virtualenv
- Wait for creation
- Reinstall packages

---

### **"Markdown checkboxes don't work"**

**Problem:** Plugin not installed or preview not enabled.

**Solution:**

1. **Install plugin:**
   - `Ctrl + Alt + S` → Plugins
   - Marketplace → Search "Markdown Navigator Enhanced"
   - Install → Restart PyCharm

2. **Enable preview:**
   - Open any `.md` file
   - Press `Ctrl + Shift + V`
   - Or: Click split preview icon (top right)

3. **Click checkboxes:**
   - Must be in **Preview mode**
   - Click directly on checkbox [ ]
   - It becomes checked [x]

---

### **"Git not configured"**

**Problem:** First time using Git in PyCharm.

**Solution:**

**Configure Git:**
```powershell
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**In PyCharm:**
- VCS → Enable Version Control Integration → Git
- First commit: `Ctrl + K` → Add message → Commit

---

### **"Directory already exists" error**

**Problem:** Running setup script multiple times.

**Solution:**

**Delete existing folders first:**
```powershell
# Remove all week folders
Remove-Item -Recurse -Force week-*

# Remove support folders
Remove-Item -Recurse -Force resources, papers, blog-drafts, portfolio

# Then run script again
.\setup.ps1
```

**Or just skip script** - folders already exist!

---

## ✅ Verification Checklist

**After setup, verify everything:**

- [ ] **Python Interpreter:** Settings shows "Python 3.12 (ml-research-12weeks)" (no "invalid")
- [ ] **Terminal:** Shows `(.venv)` at start
- [ ] **Python works:** `python --version` shows 3.12.x
- [ ] **Pip works:** `pip --version` works
- [ ] **PyTorch installed:** `python -c "import torch; print(torch.__version__)"` works
- [ ] **Folders created:** week-01 through week-12 visible in Project panel
- [ ] **Subfolders exist:** Each week has code, notebooks, experiments, notes
- [ ] **Support folders:** resources, papers, blog-drafts, portfolio visible
- [ ] **Markdown plugin:** Installed and preview works with `Ctrl + Shift + V`

**If all checked:** 🎉 **Setup complete! Start Week 1!**

---

## 🚀 Next Steps

1. **Open `week-01\README.md`** for Week 1 tasks
2. **Open `PROGRESS.md`** to track progress
3. **Start first task:** Watch Stanford CS231n Lecture 1
4. **Check off items** as you complete them
5. **Commit daily:** `Ctrl + K`

---

**Setup complete! You're ready to start the 12-week journey!** 🎯

---

*Guide for: Windows 11, PyCharm, Python 3.12, venv*  
*Path: C:\Users\adime\Documents\Career\ml-research-12weeks*
