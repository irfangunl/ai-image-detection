@echo off
chcp 65001 >nul
echo ============================================
echo AI Image Detection Project - Kurulum
echo ============================================
echo.

REM Python versiyonunu kontrol et
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadı!
    echo Lütfen Python 3.8 veya üzeri yükleyin.
    echo İndirme linki: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [✓] Python bulundu
python --version
echo.

REM pip'i güncelle
echo [1/4] pip güncelleniyor...
python -m pip install --upgrade pip
echo.

REM Virtual environment oluştur (opsiyonel)
set /p create_venv="Virtual environment oluşturulsun mu? (E/H): "
if /i "%create_venv%"=="E" (
    echo [2/4] Virtual environment oluşturuluyor...
    python -m venv venv
    echo [✓] Virtual environment oluşturuldu
    echo [!] Aktifleştirmek için: venv\Scripts\activate
    echo.
    
    echo Virtual environment aktifleştiriliyor...
    call venv\Scripts\activate.bat
) else (
    echo [2/4] Virtual environment atlanıyor...
)
echo.

REM Requirements yükle
echo [3/4] Gerekli kütüphaneler yükleniyor...
echo Bu işlem birkaç dakika sürebilir...
echo.
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [HATA] Kütüphaneler yüklenirken hata oluştu!
    echo Lütfen hata mesajlarını kontrol edin.
    pause
    exit /b 1
)

echo.
echo [4/4] Kurulum tamamlanıyor...
echo.

REM Jupyter kernel ekle
python -m ipykernel install --user --name=ai_detection --display-name "AI Detection"

echo ============================================
echo [✓] KURULUM TAMAMLANDI!
echo ============================================
echo.
echo Yüklenen Kütüphaneler:
python -m pip list | findstr "numpy pandas matplotlib seaborn opencv scikit-learn tensorflow jupyter"
echo.
echo ============================================
echo Sonraki Adımlar:
echo ============================================
echo 1. Kaggle'dan CIFAKE veri setini indirin:
echo    https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
echo.
echo 2. Veri setini 'cifake' klasörüne çıkarın
echo.
echo 3. VS Code'da AI_Image_Detection.ipynb dosyasını açın
echo.
echo 4. Notebook'u çalıştırın
echo.
echo ============================================
echo.
pause
