# Digital Image Processing Tool

A web-based application for digital image processing with various operations and analysis capabilities.

## Features

### Single Image Processing
- Basic operations and conversions
- Histogram analysis and equalization
- Image filtering and enhancement
- Noise management
- Frequency domain analysis
- Color space conversions
- Texture analysis
- Image compression

### Dual Image Processing
- Logical operations
- Histogram matching

## Tech Stack
- FastAPI (Backend)
- OpenCV, NumPy, scikit-image (Image Processing)
- HTML5, CSS3, JavaScript, Bootstrap 5 (Frontend)
- Matplotlib (Visualization)

## Quick Start

1. Clone repository:
```
git clone [your-repository-url]
cd [project-directory]
```

2. Create and activate virtual environment:
```
# Windows
python -m venv venv
.\venv\Scripts\activate
```

# macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
uvicorn main:app --reload
```

5. Open your browser and go to http://127.0.0.1:8000.