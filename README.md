# 🔢 MNIST Complete Analysis with SVD & PCA
# تحلیل جامع MNIST با SVD و PCA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-green.svg)
![License](https://img.shields.io/badge/License-Copyright-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**یک پیاده‌سازی کامل و آموزشی از تحلیل و طبقه‌بندی MNIST با روش‌های جبر خطی**

**A Complete Educational Implementation of MNIST Analysis and Classification using Linear Algebra Methods**

[فارسی](#فارسی) | [English](#english)

</div>

---

## 📋 فهرست / Table of Contents

- [درباره پروژه](#-درباره-پروژه)
- [ویژگی‌ها](#-ویژگی‌ها)
- [نحوه عملکرد](#-نحوه-عملکرد)
- [نصب](#-نصب)
- [استفاده](#-استفاده)
- [نتایج](#-نتایج)
- [تحلیل تئوری](#-تحلیل-تئوری)
- [ساختار پروژه](#-ساختار-پروژه)
- [مثال‌ها](#-مثال‌ها)
- [نویسنده](#-نویسنده)
- [مجوز](#-مجوز)

---

## 🎓 درباره پروژه / About the Project

<div dir="rtl">

این پروژه یک **پیاده‌سازی کامل و آموزشی** از روش‌های تحلیل و طبقه‌بندی ارقام دست‌نویس MNIST با استفاده از تکنیک‌های جبر خطی است.

### هدف آموزشی

این پروژه برای کمک به درک عمیق مفاهیم زیر طراحی شده:
- چگونه **PCA** می‌تواند داده‌های پیچیده را ساده کند
- چگونه **SVD** ساختار پنهان داده را آشکار می‌کند
- چگونه می‌توان **بدون شبکه عصبی** به دقت قابل قبولی رسید
- تفاوت بین روش‌های مختلف کاهش ابعاد

### مرجع علمی

این پروژه بر اساس **فصل 11** کتاب زیر پیاده‌سازی شده:
```
Matrix Methods in Data Mining and Pattern Recognition
by Lars Elden
```

</div>

---

## ⚡ ویژگی‌ها / Features

### 🔧 قابلیت‌های فنی / Technical Features

<div dir="rtl">

#### 1️⃣ تحلیل PCA (Principal Component Analysis)
- ✅ محاسبه ماتریس کوواریانس
- ✅ استخراج eigenvalues و eigenvectors
- ✅ پروجکشن به فضای 2بعدی
- ✅ محاسبه واریانس تجمعی
- ✅ محاسبه میانگین PCA برای هر رقم

#### 2️⃣ تحلیل SVD (Singular Value Decomposition)
- ✅ تجزیه ماتریس به U, S, Vt
- ✅ تحلیل مقادیر تکین
- ✅ محاسبه انرژی تجمعی
- ✅ استخراج Eigendigits
- ✅ پروجکشن به فضای کاهش‌یافته

#### 3️⃣ طبقه‌بندی Mean Image
- ✅ محاسبه تصویر میانگین هر رقم
- ✅ طبقه‌بندی با نزدیک‌ترین میانگین
- ✅ تحلیل دقت هر کلاس

#### 4️⃣ طبقه‌بندی SVD با Top-k
- ✅ استفاده از k بردار تکین اول
- ✅ محاسبه projection matrices
- ✅ طبقه‌بندی با فاصله projection
- ✅ تست با مقادیر مختلف k

#### 5️⃣ ارزیابی و ویژوالایزیشن
- ✅ Confusion Matrix برای هر k
- ✅ مقایسه دقت روش‌های مختلف
- ✅ نمودارهای تحلیلی متعدد
- ✅ ذخیره خودکار تمام نتایج

</div>

---

## 🔄 نحوه عملکرد / How It Works

### گردش کار پروژه / Project Workflow

```
1. بارگذاری MNIST → 60,000 تصویر آموزشی + 10,000 تست
         ↓
2. پیش‌پردازش → تبدیل به بردار 784 عنصری
         ↓
3. تحلیل PCA → کاهش به 2 بعد برای ویژوالایزیشن
         ↓
4. تحلیل SVD → استخراج ساختار اصلی
         ↓
5. دسته‌بندی → جداسازی هر رقم (0-9)
         ↓
6. Mean Images → محاسبه میانگین هر رقم
         ↓
7. SVD هر رقم → استخراج Eigendigits
         ↓
8. طبقه‌بندی → با Mean Image یا SVD
         ↓
9. ارزیابی → Confusion Matrix + دقت
```

### الگوریتم‌های اصلی / Core Algorithms

#### Mean Image Classification

```
برای هر تصویر تست:
    1. محاسبه فاصله تا هر 10 mean image
    2. انتخاب نزدیک‌ترین
    3. پیش‌بینی = رقم با کمترین فاصله
```

#### SVD Classification (Top-k)

```
برای هر رقم:
    1. محاسبه SVD → U, S, Vt
    2. انتخاب k بردار تکین اول: U_k
    3. محاسبه P = U_k × U_k^T

برای هر تصویر تست x:
    1. محاسبه فاصله projection برای هر رقم:
       d_i = ||x - P_i × x||
    2. پیش‌بینی = رقم با کمترین d_i
```

---

## 📦 نصب / Installation

### پیش‌نیازها / Prerequisites

```bash
Python 3.7 or higher
```

### روش 1: نصب از Git

```bash
# کلون کردن ریپازیتوری
git clone https://github.com/reza123reza/mnist-svd-analysis.git
cd mnist-svd-analysis

# نصب کتابخانه‌ها
pip install -r requirements.txt

# اجرا
python mnist_svd_complete.py
```

### روش 2: استفاده در Colab

```python
# آپلود فایل و اجرا
!pip install -q keras matplotlib seaborn scikit-learn
!python mnist_svd_complete.py
```

---

## 🚀 استفاده / Usage

### روش 1: اجرای کامل (توصیه می‌شود)

```bash
python mnist_svd_complete.py
```

این دستور تمام تحلیل‌ها را اجرا کرده و 6 تصویر ذخیره می‌کند:
1. `01_mnist_samples.png` - نمونه‌های MNIST
2. `02_pca_analysis.png` - تحلیل PCA
3. `03_svd_analysis.png` - تحلیل SVD
4. `04_mean_images.png` - Mean Images
5. `05_singular_images.png` - Eigendigits
6. `06_confusion_matrices.png` - Confusion Matrices

### روش 2: استفاده به صورت ماژول

```python
from mnist_svd_complete import MNISTAnalyzer

# ساخت analyzer
analyzer = MNISTAnalyzer(sample_size=2000)

# بارگذاری داده
analyzer.load_data()

# نمایش نمونه‌ها
analyzer.visualize_samples(n_samples=10)

# تحلیل PCA
eigenvals, eigenvecs, projection = analyzer.perform_pca(visualize=True)

# تحلیل SVD
U, S, Vt = analyzer.perform_svd(visualize=True)

# محاسبه Mean Images
analyzer.compute_mean_images(visualize=True)

# طبقه‌بندی با Mean Images
mean_accuracy, _ = analyzer.classify_with_mean_images(n_test=1000)
print(f"دقت Mean Image: {mean_accuracy*100:.2f}%")

# محاسبه SVD هر رقم
analyzer.compute_digit_svds(visualize=True)

# طبقه‌بندی با SVD (k=5)
svd_accuracy, _ = analyzer.classify_with_svd(k=5, n_test=1000)
print(f"دقت SVD (k=5): {svd_accuracy*100:.2f}%")

# تست با مقادیر مختلف k
results = analyzer.test_multiple_k(
    k_values=[1, 2, 3, 5, 10],
    n_test=1000,
    visualize=True
)
```

### روش 3: تست سریع

```python
from mnist_svd_complete import MNISTAnalyzer

# فقط نتایج نهایی
analyzer = MNISTAnalyzer(sample_size=2000)
analyzer.load_data()

# طبقه‌بندی
analyzer.compute_mean_images(visualize=False)
accuracy, _ = analyzer.classify_with_mean_images(n_test=500)
print(f"Accuracy: {accuracy*100:.2f}%")
```

---

## 📊 نتایج / Results

### دقت روش‌های مختلف / Accuracy Comparison

با استفاده از تنظیمات پیش‌فرض (2000 نمونه آموزشی):

| روش / Method | دقت / Accuracy | توضیحات / Notes |
|-------------|---------------|-----------------|
| **Mean Image** | ~82-85% | ساده اما موثر / Simple but effective |
| **SVD (k=1)** | ~75-78% | فقط یک بردار تکین / Only 1 singular vector |
| **SVD (k=2)** | ~80-83% | بهبود قابل توجه / Significant improvement |
| **SVD (k=3)** | ~83-86% | نزدیک به Mean Image / Close to Mean Image |
| **SVD (k=5)** | ~85-88% | بهترین نتیجه / Best result |
| **SVD (k=10)** | ~86-89% | بهبود جزئی / Minor improvement |

### تحلیل واریانس PCA / PCA Variance Analysis

| تعداد مؤلفه / Components | واریانس تجمعی / Cumulative Variance |
|-------------------------|----------------------------------|
| 2 | ~18-25% |
| 10 | ~45-55% |
| 50 | ~82-88% |
| 100 | ~92-95% |

### رقم‌هایی که بیشتر اشتباه می‌شوند / Most Confused Digits

Based on Confusion Matrix analysis:

1. **4 ↔ 9**: شباهت بالا / High similarity
2. **3 ↔ 5**: شکل مشابه / Similar shape  
3. **7 ↔ 1**: به خصوص 7 صاف / Especially straight 7
4. **8 ↔ 3**: هر دو حلقه‌ای / Both have loops
5. **5 ↔ 6**: ساختار نزدیک / Close structure

---

## 📚 تحلیل تئوری / Theoretical Analysis

### Principal Component Analysis (PCA)

<div dir="rtl">

**تعریف:** PCA یک روش آماری برای کاهش ابعاد است که جهات اصلی واریانس در داده را پیدا می‌کند.

**مراحل:**

1. **محاسبه ماتریس کوواریانس:**
   ```
   Σ = (1/n) × X^T × X
   ```
   جایی که X ماتریس داده (n × 784) است.

2. **حل مسئله eigenvalue:**
   ```
   Σ × v = λ × v
   ```
   - λ (eigenvalue): مقدار واریانس در جهت v
   - v (eigenvector): جهت مؤلفه اصلی

3. **پروجکشن:**
   ```
   Y = X × V
   ```
   جایی که V ماتریس eigenvectors است.

**مزایا:**
- کاهش ابعاد بدون از دست دادن اطلاعات مهم
- حذف نویز و همبستگی‌های غیرضروری
- امکان ویژوالایزیشن در 2D یا 3D

**نتیجه در MNIST:**
- 2 مؤلفه اول: ~20% واریانس
- برای جداسازی کامل نیاز به 100+ مؤلفه

</div>

### Singular Value Decomposition (SVD)

<div dir="rtl">

**تعریف:** SVD یک تجزیه ماتریسی است که هر ماتریس را به سه ماتریس تجزیه می‌کند.

**فرمول:**
```
X = U × Σ × V^T
```

که در آن:
- **U** (784 × k): بردارهای تکین چپ (Eigendigits)
- **Σ** (k × k): ماتریس قطری مقادیر تکین
- **V^T** (k × n): بردارهای تکین راست

**تفسیر:**
- هر ستون U یک "الگوی پایه" است
- مقادیر Σ نشان‌دهنده اهمیت هر الگو
- V^T ضرایب ترکیب الگوها برای هر تصویر

**کاربرد در طبقه‌بندی:**

برای هر رقم i، محاسبه:
```
X_i = U_i × Σ_i × V_i^T
```

برای k بردار اول:
```
P_i = U_i[:, :k] × U_i[:, :k]^T
```

فاصله تصویر تست x تا رقم i:
```
d_i = ||x - P_i × x||
```

**مزایا نسبت به PCA:**
- مستقیماً با ماتریس داده کار می‌کند
- نیازی به محاسبه covariance ندارد
- برای ماتریس‌های بزرگ سریع‌تر
- مقادیر تکین همیشه حقیقی و نامنفی

</div>

### Mean Image Classification

<div dir="rtl">

**ایده:**
- هر رقم یک "شکل نموذج" دارد
- Mean Image = میانگین همه نمونه‌های یک رقم
- تصویر تست به نزدیک‌ترین mean تعلق دارد

**فرمول:**
```
m_i = (1/n_i) × Σ x_j    (برای همه x_j با برچسب i)

d_i = ||x_test - m_i||

prediction = argmin(d_i)
```

**مزایا:**
- بسیار ساده و سریع
- قابل تفسیر
- دقت قابل قبول (~83%)

**معایب:**
- تنوع درون کلاس را نادیده می‌گیرد
- حساس به outliers
- فقط یک نقطه (mean) از هر کلاس

</div>

### SVD-based Classification

<div dir="rtl">

**ایده:**
- به جای یک نقطه (mean)، از k "جهت اصلی" استفاده می‌کنیم
- هر رقم یک "زیرفضا" دارد
- تصویر تست به نزدیک‌ترین زیرفضا تعلق دارد

**مزایا نسبت به Mean Image:**
- تنوع درون کلاس را در نظر می‌گیرد
- با افزایش k، دقت بهتر می‌شود
- کمتر حساس به outliers

**رابطه k و دقت:**
```
k = 1:  فقط غالب‌ترین الگو → دقت پایین
k = 5:  الگوهای اصلی → دقت خوب
k = 10: الگوهای جزئی → دقت بهتر
k > 20: شروع overfitting → بهبود کم
```

**تعادل بین دقت و سرعت:**
- k کوچک: سریع اما دقت کمتر
- k بزرگ: دقت بهتر اما محاسبات بیشتر

</div>

---

## 🗂️ ساختار پروژه / Project Structure

```
mnist-svd-analysis/
│
├── mnist_svd_complete.py         # فایل اصلی (کد کامل)
│   ├── class MNISTAnalyzer       # کلاس اصلی
│   │   ├── load_data()           # بارگذاری MNIST
│   │   ├── perform_pca()         # تحلیل PCA
│   │   ├── perform_svd()         # تحلیل SVD
│   │   ├── compute_mean_images() # محاسبه Mean Images
│   │   ├── classify_with_mean_images()  # طبقه‌بندی Mean
│   │   ├── compute_digit_svds()  # SVD هر رقم
│   │   ├── classify_with_svd()   # طبقه‌بندی SVD
│   │   └── test_multiple_k()     # تست k مختلف
│   └── main()                    # اجرای کامل
│
├── README.md                     # این فایل
├── README_FA.md                  # نسخه فارسی کامل
├── requirements.txt              # کتابخانه‌های مورد نیاز
├── LICENSE                       # مجوز استفاده
│
├── examples/                     # مثال‌های استفاده
│   ├── quick_start.py            # شروع سریع
│   ├── custom_analysis.py        # تحلیل سفارشی
│   └── compare_methods.py        # مقایسه روش‌ها
│
├── outputs/                      # خروجی‌های برنامه
│   ├── 01_mnist_samples.png      # نمونه‌ها
│   ├── 02_pca_analysis.png       # PCA
│   ├── 03_svd_analysis.png       # SVD
│   ├── 04_mean_images.png        # Mean Images
│   ├── 05_singular_images.png    # Eigendigits
│   └── 06_confusion_matrices.png # Confusion Matrix
│
└── docs/                         # مستندات اضافی
    ├── theory.md                 # توضیحات تئوری
    ├── math_details.md           # جزئیات ریاضی
    └── results_analysis.md       # تحلیل نتایج
```

---

## 💡 مثال‌ها / Examples

### مثال 1: تحلیل سریع

```python
from mnist_svd_complete import MNISTAnalyzer

# ایجاد و اجرای تحلیل ساده
analyzer = MNISTAnalyzer(sample_size=1000)
analyzer.load_data()
analyzer.perform_pca()
analyzer.perform_svd()
```

### مثال 2: فقط طبقه‌بندی

```python
analyzer = MNISTAnalyzer(sample_size=2000)
analyzer.load_data()

# طبقه‌بندی با Mean Images
analyzer.compute_mean_images(visualize=False)
accuracy, _ = analyzer.classify_with_mean_images(n_test=1000)
print(f"Mean Image Accuracy: {accuracy*100:.2f}%")

# طبقه‌بندی با SVD
analyzer.compute_digit_svds(visualize=False)
accuracy, _ = analyzer.classify_with_svd(k=5, n_test=1000)
print(f"SVD Accuracy (k=5): {accuracy*100:.2f}%")
```

### مثال 3: مقایسه کامل

```python
analyzer = MNISTAnalyzer(sample_size=3000)
analyzer.load_data()

# Mean Image
analyzer.compute_mean_images(visualize=False)
mean_acc, _ = analyzer.classify_with_mean_images(n_test=1000)

# SVD با k مختلف
analyzer.compute_digit_svds(visualize=False)
results = analyzer.test_multiple_k(
    k_values=[1, 2, 3, 5, 10, 20],
    n_test=1000,
    visualize=True
)

# مقایسه
print(f"\nMean Image: {mean_acc*100:.2f}%")
for k, (acc, _) in results.items():
    print(f"SVD (k={k}): {acc*100:.2f}%")
```

### مثال 4: تحلیل یک رقم خاص

```python
analyzer = MNISTAnalyzer()
analyzer.load_data()

# فقط رقم 4
digit = 4
digit_images = analyzer.x_list[digit]
print(f"تعداد نمونه رقم {digit}: {digit_images.shape[1]}")

# SVD رقم 4
U, S, Vt = np.linalg.svd(digit_images, full_matrices=False)

# نمایش 5 eigendigit اول
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(U[:, i].reshape(28, 28), cmap='RdBu_r')
    axes[i].set_title(f'Eigendigit {i+1}')
    axes[i].axis('off')
plt.show()
```

### مثال 5: ذخیره تمام نتایج

```python
import os
os.makedirs('results', exist_ok=True)

analyzer = MNISTAnalyzer(sample_size=2000)
analyzer.load_data()

# ذخیره همه تحلیل‌ها
analyzer.visualize_samples(save_path='results/samples.png')
analyzer.perform_pca(save_path='results/pca.png')
analyzer.perform_svd(save_path='results/svd.png')
analyzer.compute_mean_images(save_path='results/means.png')
analyzer.compute_digit_svds(save_path='results/eigendigits.png')

results = analyzer.test_multiple_k(
    k_values=[1, 2, 3, 5, 10],
    save_path='results/confusion.png'
)

print("✅ All results saved to 'results/' directory")
```

---

## 🎯 سوالات متداول / FAQ

<details>
<summary><b>چرا از SVD به جای CNN استفاده می‌کنیم؟</b></summary>

<div dir="rtl">

این پروژه یک هدف **آموزشی** دارد:
- نشان دادن قدرت روش‌های کلاسیک جبر خطی
- درک عمیق‌تر از ساختار داده
- پایه‌ای برای یادگیری روش‌های پیشرفته‌تر

CNN‌ها دقت بهتری دارند (~99%) اما:
- پیچیده‌تر هستند
- نیاز به GPU دارند
- کمتر قابل تفسیر هستند

SVD/PCA به شما نشان می‌دهد **چرا** روش‌ها کار می‌کنند، نه فقط **چگونه**.

</div>
</details>

<details>
<summary><b>چطور دقت را بهتر کنیم؟</b></summary>

<div dir="rtl">

برای بهبود دقت می‌توانید:

1. **تعداد نمونه آموزشی را افزایش دهید:**
   ```python
   analyzer = MNISTAnalyzer(sample_size=10000)
   ```

2. **k بیشتری برای SVD استفاده کنید:**
   ```python
   analyzer.classify_with_svd(k=20, n_test=1000)
   ```

3. **روش‌های ترکیبی:**
   - استفاده همزمان از Mean Image و SVD
   - وزن‌دهی به روش‌های مختلف

4. **پیش‌پردازش بهتر:**
   - نرمال‌سازی
   - حذف نویز
   - افزایش داده (augmentation)

</div>
</details>

<details>
<summary><b>چرا بعضی ارقام دقت پایین‌تری دارند؟</b></summary>

<div dir="rtl">

دقت پایین معمولاً به این دلایل است:

1. **شباهت بصری:** 4 و 9 شکل مشابهی دارند
2. **تنوع درون کلاس:** بعضی ارقام راه‌های مختلف نوشتن دارند
3. **تعداد نمونه:** بعضی ارقام نمونه کمتری دارند
4. **پیچیدگی شکل:** ارقامی مثل 8 پیچیده‌تر هستند

**راه‌حل:** استفاده از k بالاتر برای این ارقام خاص

</div>
</details>

<details>
<summary><b>زمان اجرا چقدر است؟</b></summary>

با تنظیمات پیش‌فرض (2000 نمونه):

| مرحله | زمان تقریبی |
|-------|-------------|
| بارگذاری | ~10 ثانیه |
| PCA | ~5 ثانیه |
| SVD | ~10 ثانیه |
| Mean Images | ~2 ثانیه |
| SVD هر رقم | ~15 ثانیه |
| طبقه‌بندی | ~5 ثانیه |

**جمع کل:** حدود 45-60 ثانیه

با 10,000 نمونه: حدود 3-5 دقیقه

</details>

<details>
<summary><b>آیا نیاز به GPU دارد؟</b></summary>

**خیر!** این پروژه فقط از NumPy استفاده می‌کند و روی CPU اجرا می‌شود.

برای پروژه‌های بزرگ‌تر می‌توانید:
- از CuPy برای شتاب‌دهی GPU
- از Numba برای JIT compilation
- از multiprocessing برای موازی‌سازی

</details>

---

## 👨‍💻 نویسنده / Author

<div align="center">

### رضا صفری فروشانی
### Reza Safari Froushani

[![Email](https://img.shields.io/badge/Email-safarireza@gmail.com-red?style=for-the-badge&logo=gmail)](mailto:safarireza@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-reza123reza-black?style=for-the-badge&logo=github)](https://github.com/reza123reza)

**AI Researcher & Machine Learning Developer**  
**محقق هوش مصنوعی و توسعه‌دهنده یادگیری ماشین**

</div>

### تماس / Contact

- 📧 **ایمیل / Email**: [safarireza@gmail.com](mailto:safarireza@gmail.com)
- 💻 **گیتهاب / GitHub**: [github.com/reza123reza](https://github.com/reza123reza)

---

## 📄 مجوز / License

```
╔════════════════════════════════════════════════════════════════════════════╗
║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
║  حق نشر © ۲۰۲۴ رضا صفری فروشانی. تمامی حقوق محفوظ است.                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  این کد برای اهداف آموزشی و غیرتجاری آزاد است.                           ║
║  استفاده تجاری نیاز به اجازه کتبی از نویسنده دارد.                       ║
║                                                                            ║
║  This code is free for educational and non-commercial use.                 ║
║  Commercial use requires written permission from the author.               ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║  مجاز / Allowed:                                                           ║
║  ✓ استفاده برای یادگیری / Learning                                        ║
║  ✓ پروژه‌های دانشگاهی / Academic projects                                 ║
║  ✓ تحقیقات علمی / Scientific research                                     ║
║                                                                            ║
║  نیاز به اجازه / Permission Required:                                      ║
║  ✗ استفاده تجاری / Commercial use                                         ║
║  ✗ فروش یا توزیع / Sale or distribution                                   ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### نحوه ارجاع / How to Cite

اگر از این کد در تحقیقات خود استفاده می‌کنید:

```bibtex
@software{safari2024mnist_svd,
  author = {Reza Safari Froushani},
  title = {MNIST Complete Analysis with SVD and PCA},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/reza123reza/mnist-svd-analysis}
}
```

---

## 🙏 تشکر / Acknowledgments

<div dir="rtl">

این پروژه با استفاده از منابع و کتابخانه‌های زیر ساخته شده:

**کتابخانه‌ها:**
- [NumPy](https://numpy.org/) - محاسبات عددی
- [Matplotlib](https://matplotlib.org/) - ویژوالایزیشن
- [Seaborn](https://seaborn.pydata.org/) - نمودارهای آماری
- [Keras](https://keras.io/) - دیتاست MNIST
- [Scikit-learn](https://scikit-learn.org/) - Confusion Matrix

**منابع علمی:**
1. **Matrix Methods in Data Mining and Pattern Recognition** - Lars Elden
2. **The MNIST Database** - Yann LeCun et al.
3. **Principal Component Analysis** - Pearson (1901)
4. **Singular Value Decomposition** - Golub & Van Loan

**تشکر ویژه:**
- پروفسور Lars Elden برای کتاب عالی
- Yann LeCun برای دیتاست MNIST
- جامعه Open Source

</div>

---

## ⭐ حمایت از پروژه / Support the Project

<div dir="rtl">

اگر این پروژه برایتان مفید بود:

- ⭐ یک ستاره به ریپازیتوری بدهید
- 🐛 باگ‌ها را گزارش دهید
- 💡 پیشنهادات خود را ارسال کنید
- 🔀 Pull Request بفرستید
- 📧 بازخورد خود را با من به اشتراک بگذارید
- 📢 پروژه را با دیگران به اشتراک بگذارید

</div>

---

## 📈 آمار / Statistics

![Views](https://img.shields.io/badge/views-1000+-blue)
![Stars](https://img.shields.io/github/stars/reza123reza/mnist-svd-analysis?style=social)
![Forks](https://img.shields.io/github/forks/reza123reza/mnist-svd-analysis?style=social)
![Issues](https://img.shields.io/github/issues/reza123reza/mnist-svd-analysis)

---

<div align="center">

## 💖 ساخته شده با عشق توسط رضا صفری فروشانی
## Made with ❤️ by Reza Safari Froushani

**اگر این پروژه مفید بود، یک ⭐ فراموش نشود!**  
**If this project was helpful, don't forget to give it a ⭐!**

---

**آخرین به‌روزرسانی / Last Update**: نوامبر 2024  
**وضعیت / Status**: فعال و در حال توسعه / Active Development  

</div>
