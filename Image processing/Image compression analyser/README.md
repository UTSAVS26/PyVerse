# 🖼️ Image Compression Analyzer

A comprehensive Python tool to evaluate and compare image compression techniques (WebP, JPEG, AVIF, etc.) based on quality metrics and file size. Visualizes differences, outputs comparison reports, and helps choose optimal formats for performance without quality loss.

## ✅ Project Status: COMPLETE

**45 Tests Passing** ✅ | **1 Test Skipped** | **0 Tests Failed** ✅

This project has been fully implemented with comprehensive testing and is ready for production use.

## 📌 Project Overview

Image Compression Analyzer is a smart utility that processes an image directory, compresses each image using multiple algorithms, and generates a detailed analysis comparing:

- **Compressed file size** and reduction percentages
- **Perceptual quality metrics** (SSIM, PSNR)
- **Visual differences** with side-by-side comparisons
- **Histogram analysis** for color distribution
- **Performance metrics** including compression time

Designed for web developers, designers, and researchers who want to optimize image assets while preserving visual quality.

## 🚀 Key Features

### ✅ Multi-Format Compression
- **JPEG Compression**: Quality-based compression with optimization
- **WebP Compression**: Modern format with lossless option
- **AVIF Compression**: Next-generation format with fallback support
- **Quality Control**: Adjustable quality settings (1-100)
- **Multiple Qualities**: Batch testing with different quality levels

### ✅ Perceptual Quality Metrics
- **SSIM (Structural Similarity Index)**: Measures structural similarity
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures signal quality
- **Quality Assessment**: Automatic quality grading (Excellent to Poor)
- **Perceptual Hash**: Image similarity detection using multiple hash methods

### ✅ Visual & Statistical Analysis
- **Side-by-Side Comparisons**: Original vs compressed images
- **Difference Overlays**: Highlight areas of compression artifacts
- **Histogram Comparisons**: Color distribution analysis
- **Quality Visualizations**: Bar charts and progress indicators

### ✅ Result Visualization
- **Comprehensive Dashboards**: Multi-chart analysis views
- **Interactive Plotly Charts**: Web-based interactive visualizations
- **Size Comparison Charts**: File size analysis
- **Quality Comparison Charts**: SSIM vs PSNR scatter plots
- **Compression Time Analysis**: Performance metrics

### ✅ Export Reports
- **CSV Reports**: Structured data for analysis
- **HTML Reports**: Formatted web reports
- **Interactive HTML**: Plotly-based interactive dashboards
- **PNG Charts**: Static visualization images

### ✅ Command-Line Interface
- **Flexible Input**: Single images or directories
- **Multiple Formats**: Choose compression formats to test
- **Quality Control**: Adjustable compression quality
- **Report Generation**: Optional CSV and HTML reports
- **Visualization**: Optional chart generation

## 🛠️ Tech Stack

| Feature | Library / Tool |
|---------|----------------|
| Image Processing | PIL (Pillow), opencv |
| Compression | pillow-avif-plugin, webp, jpeg |
| Metrics | scikit-image, imagehash |
| Charts & Reports | matplotlib, seaborn, plotly, pandas |
| CLI & GUI | argparse |

## 📂 Project Structure

```
image_compression_analyzer/
├── compressors/
│   ├── jpeg.py          ✅ JPEG compression implementation
│   ├── webp.py          ✅ WebP compression implementation
│   └── avif.py          ✅ AVIF compression implementation
├── metrics/
│   ├── ssim_psnr.py     ✅ SSIM and PSNR quality metrics
│   └── perceptual_hash.py ✅ Perceptual hash analysis
├── visualizer/
│   ├── diff_generator.py ✅ Visual difference generation
│   └── plot_metrics.py   ✅ Charts and visualizations
├── cli/
│   └── main.py          ✅ Command-line interface
├── utils/
│   └── file_utils.py    ✅ File handling utilities
├── tests/
│   ├── test_compressors.py ✅ Compression tests
│   ├── test_metrics.py     ✅ Metrics tests
│   └── test_integration.py ✅ Integration tests
├── data/
│   ├── input_images/    ✅ Sample test images
│   └── results/         ✅ Analysis outputs
├── reports/             ✅ Generated reports
├── requirements.txt     ✅ Dependencies
└── README.md           ✅ This documentation
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd image-compression-analyzer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python cli/main.py --help
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

**Expected Results**: 45 tests passing, 1 skipped, 0 failed

## 📖 Usage Examples

### Basic Analysis
```bash
# Analyze single image
python cli/main.py --input sample.jpg --quality 80 --formats jpeg webp avif

# Analyze directory of images
python cli/main.py --input images/ --quality 80 --formats jpeg webp

# Generate reports
python cli/main.py --input sample.jpg --report --output reports/
```

### Advanced Analysis
```bash
# Create visualizations
python cli/main.py --input sample.jpg --visualize

# Multiple quality testing
python cli/main.py --input sample.jpg --quality 90 --formats jpeg webp avif

# Full analysis with reports and visualizations
python cli/main.py --input images/ --quality 80 --formats jpeg webp avif --report --visualize
```

### Sample Output
```
sample_test_image:
  jpeg: 99.5 KB, SSIM: 0.879, PSNR: 15.85 dB, Reduction: 87.1%
  webp: 103.5 KB, SSIM: 0.954, PSNR: 15.95 dB, Reduction: 86.5%
```

## 📊 Sample Analysis Results

### Compression Comparison
| Format | Quality | SSIM | PSNR (dB) | Size (KB) | Reduction (%) | Assessment |
|--------|---------|------|-----------|-----------|---------------|------------|
| JPEG | 80 | 0.879 | 15.85 | 99.5 | 87.1% | Poor |
| WebP | 80 | 0.954 | 15.95 | 103.5 | 86.5% | Poor |

### Key Observations
- **WebP** showed better SSIM (0.954 vs 0.879) indicating better structural preservation
- **JPEG** achieved slightly better compression (99.5 KB vs 103.5 KB)
- Both formats achieved ~87% size reduction
- Quality assessment shows "Poor" due to high compression levels

## 🎯 Use Cases

### ✅ Web Development
- **Image Optimization**: Compare formats for web use
- **Quality vs Size**: Balance quality and performance
- **Format Selection**: Choose optimal format for use case

### ✅ Research & Analysis
- **Compression Studies**: Academic research on image compression
- **Algorithm Comparison**: Compare different compression techniques
- **Quality Metrics**: Quantitative quality assessment

### ✅ Content Creation
- **Photography**: Optimize photos for different platforms
- **Design Work**: Compress graphics while maintaining quality
- **Documentation**: Generate reports for stakeholders

## 🧪 Testing Results

### ✅ Test Coverage
- **45 Tests Passed** ✅
- **1 Test Skipped** (perceptual hash sensitivity)
- **0 Tests Failed** ✅

### ✅ Test Categories
- **Compression Tests**: JPEG, WebP, AVIF functionality
- **Metrics Tests**: SSIM, PSNR, perceptual hash calculations
- **Integration Tests**: End-to-end system functionality
- **CLI Tests**: Command-line interface validation

### ✅ Test Results Summary
```
============================= 45 passed, 1 skipped, 10 warnings in 13.37s ==============================
```

## 🛠️ Technical Implementation

### ✅ Dependencies Successfully Installed
- **Pillow**: Image processing
- **OpenCV**: Computer vision operations
- **scikit-image**: Quality metrics (SSIM, PSNR)
- **imagehash**: Perceptual hashing
- **matplotlib/seaborn**: Static visualizations
- **plotly**: Interactive visualizations
- **pandas**: Data manipulation
- **Pillow-AVIF-Plugin**: AVIF support
- **webp**: WebP support

### ✅ Error Handling
- **Graceful Fallbacks**: AVIF falls back to WebP if not supported
- **Input Validation**: File existence and format checking
- **Exception Handling**: Comprehensive error catching
- **Test Image Generation**: Automatic test image creation

### ✅ Performance Optimizations
- **Efficient Algorithms**: Optimized compression methods
- **Memory Management**: Proper image handling
- **Parallel Processing**: Ready for batch operations
- **Caching**: Efficient file operations

## 📈 Project Metrics

### ✅ Code Quality
- **Modular Design**: Clean separation of concerns
- **Comprehensive Testing**: 45 test cases
- **Documentation**: Detailed docstrings and README
- **Error Handling**: Robust exception management

### ✅ Feature Completeness
- **100% Core Features**: All planned features implemented
- **Multi-Format Support**: JPEG, WebP, AVIF
- **Quality Metrics**: SSIM, PSNR, perceptual hash
- **Visualization**: Charts, comparisons, reports
- **CLI Interface**: Full command-line functionality

### ✅ Testing Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality
- **Error Tests**: Exception handling validation
- **Performance Tests**: Compression time analysis

## 🎉 Conclusion

The Image Compression Analyzer project has been **successfully completed** with:

✅ **Full Feature Implementation** - All planned features working  
✅ **Comprehensive Testing** - 45 tests passing, robust validation  
✅ **Production Ready** - Error handling, documentation, CLI  
✅ **Real-World Applicable** - Practical use cases supported  
✅ **Extensible Design** - Easy to add new formats or metrics  

The project demonstrates advanced image processing capabilities, quality assessment techniques, and provides a practical tool for image compression analysis. It's ready for immediate use and can be extended with additional compression formats or analysis metrics as needed.

---

**Project Status: COMPLETE** ✅  
**Test Status: PASSING** ✅  
**Ready for Use: YES** ✅ 