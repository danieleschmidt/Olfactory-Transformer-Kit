---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior
A clear and concise description of what actually happened.

## 📱 Environment
**System Information:**
- OS: [e.g. Ubuntu 22.04, macOS 13.0, Windows 11]
- Python Version: [e.g. 3.9.7]
- Olfactory Transformer Version: [e.g. 0.1.0]
- PyTorch Version: [e.g. 2.0.1]

**Hardware:**
- CPU: [e.g. Intel i7-9700K, Apple M1]
- GPU: [e.g. NVIDIA RTX 3080, None]
- RAM: [e.g. 16GB]

**Dependencies:**
- RDKit installed: [Yes/No]
- Sensor hardware: [e.g. TGS2600, None]

## 📋 Installation Method
- [ ] pip install
- [ ] conda install  
- [ ] Docker
- [ ] From source
- [ ] Other: ___________

## 📜 Code to Reproduce
```python
# Minimal code example that reproduces the issue
from olfactory_transformer import OlfactoryTransformer

# Your code here...
```

## 📄 Error Output
```
Paste the complete error message and stack trace here
```

## 📊 Additional Context
**Input Data:**
- SMILES strings used: [e.g. "CCO", "invalid_smiles"]
- Sensor readings: [if applicable]
- Model path: [if using custom model]

**Configuration:**
```python
# Any relevant configuration settings
config = {
    "batch_size": 32,
    "device": "cuda",
    # etc...
}
```

**Screenshots/Plots:**
If applicable, add screenshots or plots to help explain your problem.

## 🔍 Debugging Information
**Log Output:**
```
Paste relevant log output here (set LOG_LEVEL=DEBUG for more detail)
```

**Memory Usage:**
- Available RAM during error: [e.g. 8GB free]
- GPU memory usage: [e.g. 6GB/8GB used]

**Network/Sensor Information:**
- Sensor connection: [e.g. /dev/ttyUSB0, mock mode]
- Network connectivity: [if using distributed features]

## 🎯 Impact
**Severity:** 
- [ ] Critical - Complete failure, no workaround
- [ ] High - Major functionality broken
- [ ] Medium - Some functionality affected
- [ ] Low - Minor issue or cosmetic

**Use Case:**
- [ ] Research/Academic
- [ ] Production/Commercial
- [ ] Development/Testing
- [ ] Learning/Tutorial

## 💡 Potential Solution
If you have ideas about what might be causing the issue or how to fix it, please describe them here.

## ✅ Checklist
Before submitting, please check:
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have included all relevant information
- [ ] I have provided a minimal code example
- [ ] I have included the complete error message
- [ ] I have specified my environment details
- [ ] I have tried with the latest version

## 🔗 Related Issues
Link any related issues or discussions:
- Related to #123
- Similar to #456

---
**Additional Notes:**
Add any other context about the problem here.