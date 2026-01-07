"""
AI Image Detection Project
Yapay Zeka Tarafından Oluşturulmuş Görsel Tespit Projesi

Bu projede, bir görselin gerçek mi yoksa yapay zeka tarafından mı oluşturulduğunu 
tespit eden bir sinir ağı modeli geliştirilmiştir.

Video Analizi: Videolardan 5 farklı frame çıkarılarak her biri ayrı ayrı analiz 
edilecek ve sonuçlar birleştirilerek karar verilecektir.
"""

# ============================================
# 1. Gerekli Kütüphanelerin Yüklenmesi
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dosya seçme için
from tkinter import Tk, filedialog

# Keras/TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow Versiyonu: {tf.__version__}")
print(f"GPU Kullanılabilir mi: {tf.config.list_physical_devices('GPU')}")


# ============================================
# 2. Veri Setinin Hazırlanması
# ============================================

"""
Veri Seti: CIFAKE - Real and AI-Generated Synthetic Images (Kaggle)

Bu veri seti:
- 60,000 gerçek görsel (CIFAR-10'dan)
- 60,000 yapay zeka üretimi görsel (Stable Diffusion v1.4)
- Her görsel 32x32 piksel boyutunda

Link: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
"""

# Veri yolu ayarları (Kaggle'dan indirdikten sonra güncelleyin)
DATA_PATH = 'cifake/'  # Veri setinin bulunduğu klasör
IMG_SIZE = 64  # Görselleri yeniden boyutlandırma
BATCH_SIZE = 32

# Eğitim modunu kullanıcıdan al
print("\n" + "="*50)
print("EĞİTİM MODU SEÇİMİ")
print("="*50)
print("1. Demo Eğitim - Hızlı test için (Her sınıftan 1000 görsel, ~5 epoch)")
print("2. Tam Eğitim - Yüksek performans için (Her sınıftan 10000 görsel, ~20 epoch)")
print("="*50)

while True:
    mode = input("\nSeçiminiz (1 veya 2): ").strip()
    if mode in ['1', '2']:
        break
    print("Hatalı seçim! Lütfen 1 veya 2 girin.")

if mode == '1':
    SAMPLE_SIZE = 1000
    EPOCHS = 10
    print("\n✓ Demo Eğitim modu seçildi")
    print(f"  - Her sınıftan {SAMPLE_SIZE} görsel kullanılacak")
    print(f"  - {EPOCHS} epoch eğitim yapılacak")
    print(f"  - Tahmini süre: 5-10 dakika\n")
else:
    SAMPLE_SIZE = 10000
    EPOCHS = 20
    print("\n✓ Tam Eğitim modu seçildi")
    print(f"  - Her sınıftan {SAMPLE_SIZE} görsel kullanılacak")
    print(f"  - {EPOCHS} epoch eğitim yapılacak")
    print(f"  - Tahmini süre: 30-60 dakika\n")


# ============================================
# 3. Görsellerin Yüklenmesi ve Ön İşleme
# ============================================

def load_dataset(data_path, img_size=64, sample_size=10000):
    """
    Veri setini yükler ve ön işleme yapar
    
    Args:
        data_path: Veri setinin yolu
        img_size: Görsellerin boyutu
        sample_size: Her sınıftan kaç görsel alınacak (hız için)
    
    Returns:
        X, y: Görsel verileri ve etiketleri
    """
    X = []
    y = []
    
    # Gerçek görseller (label = 0)
    real_path = os.path.join(data_path, 'train', 'REAL')
    if os.path.exists(real_path):
        real_images = os.listdir(real_path)[:sample_size]
        print(f"Gerçek görseller yükleniyor: {len(real_images)} adet")
        
        for img_name in real_images:
            img_path = os.path.join(real_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(0)  # Gerçek = 0
    
    # AI üretimi görseller (label = 1)
    fake_path = os.path.join(data_path, 'train', 'FAKE')
    if os.path.exists(fake_path):
        fake_images = os.listdir(fake_path)[:sample_size]
        print(f"AI üretimi görseller yükleniyor: {len(fake_images)} adet")
        
        for img_name in fake_images:
            img_path = os.path.join(fake_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(1)  # AI üretimi = 1
    
    X = np.array(X, dtype='float32') / 255.0  # Normalizasyon
    y = np.array(y)
    
    print(f"\nToplam görsel sayısı: {len(X)}")
    print(f"Gerçek görseller: {np.sum(y == 0)}")
    print(f"AI üretimi görseller: {np.sum(y == 1)}")
    
    return X, y


# ============================================
# 4. CNN Modelinin Oluşturulması
# ============================================

def create_model(input_shape=(64, 64, 3)):
    """
    AI görsel tespit modeli oluşturur
    
    Mimari:
    - 3 Convolutional blok (Conv2D + BatchNorm + MaxPooling + Dropout)
    - Flatten
    - 2 Dense katman
    - Binary classification (sigmoid)
    """
    model = Sequential([
        # İlk Conv Bloğu
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # İkinci Conv Bloğu
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Üçüncü Conv Bloğu
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Katmanlar
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Model derleme
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# 5. Görsel Tahmin Fonksiyonu
# ============================================

def predict_single_image(image_path, model, img_size=64):
    """
    Tek bir görsel için tahmin yapar
    
    Args:
        image_path: Görsel dosya yolu
        model: Eğitilmiş model
        img_size: Görsel boyutu
    
    Returns:
        prediction: Tahmin (0: Gerçek, 1: AI)
        confidence: Güven skoru
    """
    # Görseli oku
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Yeniden boyutlandır ve normalize et
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    
    # Tahmin
    pred_prob = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0][0]
    prediction = 1 if pred_prob > 0.5 else 0
    confidence = pred_prob if prediction else 1 - pred_prob
    
    # Görselleştirme
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    title = f"Tahmin: {'AI Üretimi' if prediction else 'Gerçek Görsel'}\n"
    title += f"Güven Skoru: %{confidence*100:.1f}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    return prediction, confidence


# ============================================
# 6. Ana Eğitim ve Değerlendirme Fonksiyonu
# ============================================

def train_and_evaluate():
    """
    Model eğitimi ve değerlendirmesini gerçekleştirir
    """
    print("\n" + "="*50)
    print("VERİ SETİ YÜKLENİYOR")
    print("="*50)
    
    # Veri setini yükle
    X, y = load_dataset(DATA_PATH, img_size=IMG_SIZE, sample_size=SAMPLE_SIZE)
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nEğitim seti: {X_train.shape}")
    print(f"Test seti: {X_test.shape}")
    
    # Veri görselleştirme
    print("\nÖrnek görseller görselleştiriliyor...")
    plt.figure(figsize=(15, 6))
    
    # Gerçek görseller
    for i in range(5):
        idx = np.where(y_train == 0)[0][i]
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[idx])
        plt.title('Gerçek Görsel', fontsize=10)
        plt.axis('off')
    
    # AI üretimi görseller
    for i in range(5):
        idx = np.where(y_train == 1)[0][i]
        plt.subplot(2, 5, i+6)
        plt.imshow(X_train[idx])
        plt.title('AI Üretimi', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model oluşturma
    print("\n" + "="*50)
    print("MODEL OLUŞTURULUYOR")
    print("="*50)
    
    model = create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary()
    
    # Data augmentation
    print("\nData augmentation hazırlanıyor...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train)
    
    # Model eğitimi
    print("\n" + "="*50)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("="*50)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Eğitim grafiklerini görselleştirme
    print("\nEğitim grafikleri oluşturuluyor...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss grafiği
    axes[0].plot(history.history['loss'], label='Eğitim Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validasyon Loss', linewidth=2)
    axes[0].set_title('Model Loss Grafiği', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(history.history['accuracy'], label='Eğitim Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validasyon Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy Grafiği', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model değerlendirmesi
    print("\n" + "="*50)
    print("MODEL DEĞERLENDİRMESİ")
    print("="*50)
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANS METRİKLERİ")
    print("="*50)
    print(f"Accuracy  (Doğruluk): {accuracy*100:.2f}%")
    print(f"Precision (Kesinlik): {precision*100:.2f}%")
    print(f"Recall    (Duyarlılık): {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print("="*50)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Gerçek', 'AI Üretimi'],
                yticklabels=['Gerçek', 'AI Üretimi'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Gerçek', 'AI Üretimi']))
    
    # Örnek tahminler
    print("\nÖrnek tahminler görselleştiriliyor...")
    plt.figure(figsize=(15, 10))
    
    for i in range(15):
        idx = np.random.randint(0, len(X_test))
        img = X_test[idx]
        true_label = y_test[idx]
        
        # Tahmin
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = 1 if pred_prob > 0.5 else 0
        
        # Görselleştirme
        plt.subplot(3, 5, i+1)
        plt.imshow(img)
        
        # Başlık rengi (doğru: yeşil, yanlış: kırmızı)
        color = 'green' if pred_label == true_label else 'red'
        
        title = f"Gerçek: {'AI' if true_label else 'Gerçek'}\n"
        title += f"Tahmin: {'AI' if pred_label else 'Gerçek'}\n"
        title += f"(%{pred_prob*100:.1f})"
        
        plt.title(title, fontsize=8, color=color, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Modeli kaydetme
    print("\nModel kaydediliyor...")
    model.save('ai_image_detector.h5')
    print("Model 'ai_image_detector.h5' olarak kaydedildi.")
    
    return model


# ============================================
# 7. Ana Program
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("AI IMAGE DETECTION PROJECT")
    print("Yapay Zeka Görsel Tespit Projesi")
    print("="*50)
    
    # Model eğitimi ve değerlendirmesi
    model = train_and_evaluate()
    
    print("\n" + "="*50)
    print("EĞİTİM TAMAMLANDI!")
    print("="*50)
    
    # Kullanıcıdan görsel seçmesini iste
    print("\n" + "="*50)
    print("GÖRSEL TEST MODU")
    print("="*50)
    print("\nŞimdi kendi görselinizi test edebilirsiniz!")
    print("Dosya seçme penceresi açılacak...")
    print("Veya 'q' yazarak çıkış yapabilirsiniz.")
    print("="*50)
    
    while True:
        choice = input("\nGörsel seçmek için ENTER'a basın (veya 'q' ile çıkış): ").strip().lower()
        
        if choice == 'q':
            print("\nProgram sonlandırılıyor...")
            break
        
        # Dosya seçme penceresini aç
        print("\n📂 Dosya seçme penceresi açılıyor...")
        root = Tk()
        root.withdraw()  # Ana pencereyi gizle
        root.attributes('-topmost', True)  # Pencereyi en üste getir
        
        image_path = filedialog.askopenfilename(
            title="Test etmek istediğiniz görseli seçin",
            filetypes=[
                ("Görsel Dosyaları", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        root.destroy()
        
        # Kullanıcı iptal etti
        if not image_path:
            print("❌ Dosya seçilmedi.")
            continue
        
        if not os.path.exists(image_path):
            print(f"\n❌ HATA: Görsel bulunamadı: {image_path}")
            print("Lütfen geçerli bir dosya yolu girin.")
            continue
        
        try:
            print(f"\n🔍 Görsel analiz ediliyor: {image_path}")
            print("-" * 50)
            
            prediction, confidence = predict_single_image(image_path, model, img_size=IMG_SIZE)
            
            print("\n" + "="*50)
            print("SONUÇ")
            print("="*50)
            if prediction == 0:
                print(f"✅ Bu görsel GERÇEK bir görsel")
            else:
                print(f"🤖 Bu görsel AI TARAFINDAN ÜRETİLMİŞ")
            print(f"Güven Skoru: %{confidence*100:.2f}")
            print("="*50)
            
            # Başka görsel test etmek isteyip istemediğini sor
            another = input("\nBaşka bir görsel test etmek ister misiniz? (E/H): ").strip().lower()
            if another != 'e':
                print("\nProgram sonlandırılıyor...")
                break
                
        except Exception as e:
            print(f"\n❌ HATA: Görsel işlenirken bir hata oluştu: {str(e)}")
            print("Lütfen başka bir görsel deneyin.")
    
    print("\n" + "="*50)
    print("Teşekkürler! 👋")
    print("="*50)

