# Project Setup & Run

This project is designed to be **zero-setup**.

## ✅ Requirements

* **Windows**
* **Python 3.12+** installed and available in PATH

## ▶️ How to Run

Simply double-click or execute:

```bat
run.bat
```

That’s it.

## 🔧 What `run.bat` Does

The script automatically:

1. Creates a Python virtual environment (`.venv`) **only if it doesn’t already exist**
2. Activates the virtual environment
3. Installs all dependencies from `requirements.txt`
4. Runs the application (`run.py`)

No manual setup is required.

## 📝 Notes

* The virtual environment is reused on subsequent runs
* If dependencies change, update `requirements.txt` and re-run `run.bat`
* Ensure you have the correct python version installed and the data is in the correct format and folder

---

Happy coding 🚀
