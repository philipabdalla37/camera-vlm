# Stop on errors
set -e

echo "Creating virtual environment..."
python -m venv myenv

echo "Activating virtual environment..."
source myenv/Scripts/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing project dependencies..."
python -m pip install -r requirements.txt

echo "Environment setup complete."
echo "To activate it later run:"
echo "source venv/Scripts/activate"