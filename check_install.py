import sys
print(f"Python version: {sys.version}")

packages = ['numpy', 'pandas', 'sklearn', 'fastapi', 'pydantic', 'joblib', 'mlflow']

for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(package)
            version = mod.__version__
        print(f"✅ {package}: {version}")
    except ImportError as e:
        print(f"❌ {package}: NOT INSTALLED - {e}")