# 🚀 START HERE - ML Research 12-Week Plan

**Your Setup:**
- Windows 11 Lenovo
- PyCharm with Python 3.12
- Virtual Environment (venv)
- Path: `C:\Users\adime\Documents\ml-research-12weeks`

**Time to Start:** 20 minutes

---

## 📂 Your Files (6 Core Files Only)

1. **START_HERE.md** ⭐ **YOU ARE HERE**
2. **SETUP_GUIDE.md** → Complete setup instructions
3. **setup.ps1** → PowerShell script to create folders
4. **PROGRESS.md** → Daily tracker with checkboxes
5. **WEEK_01_TASKS.md** → Week 1 daily tasks (move to `week-01\README.md` after setup)
6. **COMPLETE_12_WEEK_PLAN.md** → Full plan reference

**That's it! Only 6 files you need.**

---

## ⚡ Quick Start (20 minutes)

### **Step 1: Create Folder Structure (2 min)**

**In PyCharm Terminal:**
```powershell
# Run the setup script
.\setup.ps1
```

**If execution policy error:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

**Manual alternative:** Open SETUP_GUIDE.md → Section "Manual Folder Creation"

---

### **Step 2: Fix Python Interpreter (3 min)**

**In PyCharm:**
1. **Ctrl + Alt + S** (Settings)
2. **Project → Python Interpreter**
3. **Gear icon → Show All**
4. **Delete any "invalid" interpreters**
5. **Gear icon → Add → Virtualenv Environment → New**
6. **Base interpreter:** Browse to Python 3.13:
   ```
   C:\Users\adime\AppData\Local\Programs\Python\Python313\python.exe
   ```
7. **Location:** Should auto-fill to `.venv`
8. **OK → Wait 30 seconds**

**Verify in Terminal:**
```powershell
python --version
# Should show: Python 3.13.x
```

---

### **Step 3: Install Python Packages (10 min)**

**Copy-paste these into PyCharm Terminal:**

```powershell
# PyTorch (takes ~3 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core ML packages
pip install numpy scipy pandas scikit-learn matplotlib seaborn

# Computer Vision
pip install opencv-python albumentations timm ultralytics

# Time Series
pip install statsmodels prophet neuralprophet

# Tools
pip install wandb jupyterlab black pytest ipykernel

# Utilities
pip install gradio streamlit tqdm pillow

# Save environment
pip freeze > requirements.txt
```

**Wait for all to complete (5-10 min).**

---

### **Step 4: Install PyCharm Plugin (3 min)**

**For clickable checkboxes in markdown:**

1. **Ctrl + Alt + S** (Settings)
2. **Plugins → Marketplace**
3. Search: **"Markdown Navigator Enhanced"**
4. **Install → Restart PyCharm**

---

### **Step 5: Organize Files (2 min)**

**Move Week 1 file:**
1. Take **WEEK_01_TASKS.md**
2. Move it to: `week-01\README.md`
3. Delete the original

**Your structure should now be:**
```
ml-research-12weeks\
├── START_HERE.md
├── SETUP_GUIDE.md
├── PROGRESS.md
├── COMPLETE_12_WEEK_PLAN.md
├── setup.ps1
├── week-01\
│   └── README.md (your Week 1 tasks)
├── week-02\ through week-12\
├── resources\
├── papers\
├── blog-drafts\
└── portfolio\
```

---

### **Step 6: Start Week 1! (NOW!)**

**In PyCharm:**
1. Open **two files side-by-side:**
   - `week-01\README.md` (your tasks)
   - `PROGRESS.md` (your tracker)

2. **Split view:**
   - Right-click tab → Split Right
   - Keep both open while working

3. **Start Monday's first task:**
   - Watch Stanford CS231n Lecture 1
   - Link: http://cs231n.stanford.edu/

4. **Check off tasks as you complete them:**
   - **Ctrl + Shift + V** (markdown preview)
   - Click checkboxes!

---

## 📋 Daily Workflow

**Every Morning:**
```
1. Open PyCharm
2. Check week-XX\README.md for today's tasks
3. Review PROGRESS.md
4. Start first task
```

**During Work:**
```
1. Code in week-XX\code\
2. Experiments in week-XX\notebooks\
3. Check off completed items
4. Commit: Ctrl + K
```

**Every Evening:**
```
1. Update PROGRESS.md
2. Commit & push: Ctrl + K → Ctrl + Shift + K
3. Write reflection in week-XX\notes\
4. Preview tomorrow's tasks
```

---

## 🔧 Windows Keyboard Shortcuts

**Essential PyCharm shortcuts:**
- **Ctrl + Alt + S** → Settings
- **Ctrl + K** → Git Commit
- **Ctrl + Shift + K** → Git Push
- **Ctrl + Shift + V** → Markdown Preview
- **Alt + F12** → Terminal
- **Ctrl + /** → Comment/Uncomment
- **Ctrl + Shift + F10** → Run current file

---

## ✅ Success Checklist

Before starting Week 1:
- [ ] PyCharm project open at `C:\Users\adime\Documents\Career\ml-research-12weeks`
- [ ] Folder structure created (week-01 through week-12 visible)
- [ ] Python interpreter shows "Python 3.13 (ml-research-12weeks)"
- [ ] Terminal shows `(.venv)` at start
- [ ] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] Markdown Navigator plugin installed
- [ ] `week-01\README.md` open with tasks visible
- [ ] `PROGRESS.md` open for tracking

---

## 🎯 The 12-Week Plan Overview

**Weeks 1-2:** Computer Vision Foundations
- Classical CV, CNNs, ResNet, ViT, EfficientNet

**Weeks 3-4:** CV Applications
- Object detection (YOLO), segmentation (U-Net), medical imaging

**Weeks 5-6:** Time Series Analysis
- LSTM, Transformers (Informer), forecasting, anomaly detection

**Weeks 7-8:** Adversarial Machine Learning
- Attacks (FGSM, PGD), defenses, certified robustness

**Weeks 9-10:** Paper Reproduction + Novel Contribution
- Implement SOTA paper, add your improvement

**Week 11:** Original Research Project
- 4-6 page paper, experiments, results, arXiv submission

**Week 12:** Portfolio Polish & Job Applications
- GitHub cleanup, blog posts, resume, 20-30 applications

**Target:** ML Research Engineer roles in Netherlands/Germany/UK with visa sponsorship

---

## 🆘 Common Issues

**"Invalid Python interpreter":**
→ Read SETUP_GUIDE.md → Section "Fix Invalid Interpreter"

**"Spaces in path error":**
→ Your current path is fine (no spaces after "Career")

**"Package installation fails":**
→ Check you're in venv: Look for `(.venv)` in terminal

**"PowerShell script won't run":**
→ `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**"Can't find Python 3.13":**
→ In PowerShell: `where.exe python`

---

## 📚 Need More Details?

**Read these files:**

| What You Need | File to Read |
|---------------|--------------|
| Detailed setup steps | SETUP_GUIDE.md |
| Daily/weekly tasks | week-01\README.md (and week-02, etc.) |
| Track your progress | PROGRESS.md |
| All resources & papers | COMPLETE_12_WEEK_PLAN.md |
| Troubleshooting | SETUP_GUIDE.md → Troubleshooting section |

---

## 🎯 Right Now

**If you haven't set up yet:**
1. Read SETUP_GUIDE.md
2. Follow Steps 1-5 above
3. Open week-01\README.md
4. Start!

**If you're already set up:**
1. Open week-01\README.md
2. Start Monday's tasks
3. Watch CS231n Lecture 1
4. Begin coding!

---

**Everything is ready for Windows 11 Lenovo with venv. Let's start building your ML portfolio!** 🚀

---

*Path: C:\Users\adime\Documents\Career\ml-research-12weeks*  
*Python: 3.13 with venv*  
*IDE: PyCharm*
