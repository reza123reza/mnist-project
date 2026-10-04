"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  MNIST Digit Classification with SVD & PCA                 ║
║              طبقه‌بندی ارقام دست‌نویس MNIST با SVD و PCA                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  نویسنده: رضا صفری فروشانی (Reza Safari Froushani)                      ║
║  ایمیل: safarireza@gmail.com                                              ║
║  گیتهاب: https://github.com/reza123reza                                   ║
║  تاریخ: نوامبر 2024                                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  توضیحات:                                                                  ║
║  پیاده‌سازی کامل الگوریتم‌های کاهش ابعاد و طبقه‌بندی برای MNIST:        ║
║                                                                            ║
║  ✓ Principal Component Analysis (PCA)                                     ║
║  ✓ Singular Value Decomposition (SVD)                                     ║
║  ✓ Mean Image Classification                                              ║
║  ✓ SVD-based Classification (Top-k Singular Vectors)                      ║
║  ✓ Confusion Matrix Analysis                                              ║
║  ✓ Complete Visualization Suite                                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║  مرجع علمی:                                                               ║
║  Matrix Methods in Data Mining and Pattern Recognition                     ║
║  by Lars Elden, Chapter 11                                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# تنظیمات نمایش فارسی
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100


class MNISTAnalyzer:
    """
    کلاس جامع برای تحلیل و طبقه‌بندی MNIST با SVD و PCA
    
    این کلاس شامل تمام متدهای مورد نیاز برای:
    - بارگذاری و پیش‌پردازش MNIST
    - تحلیل PCA با ویژوالایزیشن
    - تحلیل SVD و استخراج Eigendigits  
    - طبقه‌بندی با Mean Images
    - طبقه‌بندی با Top-k Singular Vectors
    - محاسبه و نمایش Confusion Matrix
    
    نویسنده: رضا صفری فروشانی
    ایمیل: safarireza@gmail.com
    """
    
    def __init__(self, sample_size: int = 2000):
        """
        مقداردهی اولیه
        
        Args:
            sample_size: تعداد نمونه‌های استفاده شده برای آموزش
        """
        self.sample_size = sample_size
        
        # متغیرهای داده
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
        # فرمت‌های مختلف داده
        self.x_train_rowvector = None
        self.x_train_colvector = None
        self.x_test_rowvector = None
        
        # داده‌های نمونه
        self.x_train_sample = None
        self.y_train_sample = None
        
        # متغیرهای PCA
        self.pca_eigvals = None
        self.pca_eigvecs = None
        self.pca_projection = None
        
        # متغیرهای SVD
        self.svd_U = None
        self.svd_S = None
        self.svd_Vt = None
        
        # داده‌های دسته‌بندی شده
        self.x_list = []  # لیست تصاویر هر رقم
        self.y_list = []  # لیست برچسب‌های هر رقم
        
        # Mean images
        self.mean_images = []
        
        # SVD برای هر رقم
        self.digit_svds = {}  # {digit: (U, S, Vt)}
        
        self._print_header()
    
    def _print_header(self):
        """چاپ هدر برنامه"""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "MNIST SVD & PCA Complete Analysis" + " " * 29 + "║")
        print("║" + " " * 78 + "║")
        print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
        print("║" + "  ایمیل: safarireza@gmail.com".ljust(77) + " ║")
        print("║" + "  گیتهاب: https://github.com/reza123reza".ljust(77) + " ║")
        print("╚" + "═" * 78 + "╝\n")
    
    def load_data(self) -> bool:
        """
        بارگذاری دیتاست MNIST
        
        Returns:
            True در صورت موفقیت
        """
        print("═" * 80)
        print("📥 بارگذاری دیتاست MNIST")
        print("Loading MNIST Dataset")
        print("═" * 80)
        
        try:
            from keras.datasets import mnist
            
            print("\n[1/5] دانلود دیتاست...")
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            print(f"      ✓ آموزش: {len(self.x_train):,} تصویر")
            print(f"      ✓ تست: {len(self.x_test):,} تصویر")
            
            print("\n[2/5] تبدیل به بردار...")
            # Row vectors: هر سطر یک تصویر
            self.x_train_rowvector = self.x_train.reshape(len(self.x_train), -1)
            self.x_test_rowvector = self.x_test.reshape(len(self.x_test), -1)
            
            # Column vectors: هر ستون یک تصویر
            self.x_train_colvector = self.x_train_rowvector.T
            
            print(f"      ✓ Row vector: {self.x_train_rowvector.shape}")
            print(f"      ✓ Column vector: {self.x_train_colvector.shape}")
            
            print("\n[3/5] انتخاب نمونه...")
            self.x_train_sample = self.x_train_colvector[:, :self.sample_size]
            self.y_train_sample = self.y_train[:self.sample_size]
            print(f"      ✓ استفاده از {self.sample_size} نمونه")
            
            print("\n[4/5] دسته‌بندی بر اساس برچسب...")
            self._organize_by_label()
            print("      ✓ داده‌ها به 10 دسته تقسیم شدند")
            
            print("\n[5/5] آماده‌سازی کامل")
            print("      ✓ داده آماده پردازش است")
            
            print("\n" + "═" * 80)
            print("✅ بارگذاری با موفقیت انجام شد!")
            print("═" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ خطا: {str(e)}")
            return False
    
    def _organize_by_label(self):
        """دسته‌بندی داده‌ها بر اساس برچسب (0-9)"""
        self.x_list = []
        self.y_list = []
        
        # فقط sample_size نمونه‌ی اول (همون نمونه‌ای که PCA/SVD روش انجام می‌شه)
        # استفاده می‌شه - قبلاً کل ۶۰۰۰۰ تصویر آموزشی استفاده می‌شد و
        # پارامتر sample_size روی طبقه‌بندی هیچ اثری نداشت.
        for digit in range(10):
            # پیدا کردن همه نمونه‌های این رقم
            indices = np.where(self.y_train_sample == digit)[0]
            
            # استخراج تصاویر مربوطه (float برای محاسبات دقیق‌تر)
            digit_images = self.x_train_sample[:, indices].astype(np.float64)
            digit_labels = self.y_train_sample[indices]
            
            self.x_list.append(digit_images)
            self.y_list.append(digit_labels)
    
    def visualize_samples(self, n_samples: int = 10, 
                         save_path: Optional[str] = None):
        """
        نمایش نمونه‌هایی از تصاویر MNIST
        
        Args:
            n_samples: تعداد نمونه‌های نمایشی
            save_path: مسیر ذخیره (اختیاری)
        """
        print(f"\n{'═'*80}")
        print(f"🖼️  نمایش {n_samples} نمونه از دیتاست")
        print(f"{'═'*80}")
        
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        fig.suptitle('نمونه‌های دیتاست MNIST\nMNIST Dataset Samples',
                    fontsize=14, fontweight='bold', y=1.02)
        
        for idx in range(n_samples):
            row = idx // 5
            col = idx % 5
            
            # دریافت تصویر از column vector
            image = self.x_train_colvector[:, idx].reshape(28, 28)
            label = self.y_train[idx]
            
            axes[row, col].imshow(image, cmap='Greys')
            axes[row, col].set_title(f'Label: {label}', 
                                    fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ذخیره شد: {save_path}")
        
        plt.show()
        print("✓ نمایش نمونه‌ها انجام شد")
    
    def perform_pca(self, visualize: bool = True,
                   save_path: Optional[str] = None) -> Tuple:
        """
        انجام تحلیل PCA
        
        Args:
            visualize: نمایش نتایج
            save_path: مسیر ذخیره
            
        Returns:
            (eigenvalues, eigenvectors, projection)
        """
        print(f"\n{'═'*80}")
        print("📊 تحلیل مؤلفه‌های اصلی (PCA)")
        print("Principal Component Analysis")
        print(f"{'═'*80}")
        
        print("\n[1/5] محاسبه ماتریس کوواریانس...")
        covariance = np.cov(self.x_train_sample)
        print(f"      ✓ شکل: {covariance.shape}")
        
        print("\n[2/5] محاسبه eigenvalues و eigenvectors...")
        # ماتریس کوواریانس متقارنه، پس eigh (سریع‌تر و همیشه حقیقی) به‌جای eig
        self.pca_eigvals, self.pca_eigvecs = np.linalg.eigh(covariance)
        
        # مرتب‌سازی
        idx = self.pca_eigvals.argsort()[::-1]
        self.pca_eigvals = self.pca_eigvals[idx]
        self.pca_eigvecs = self.pca_eigvecs[:, idx]
        
        print(f"      ✓ تعداد eigenvalues: {len(self.pca_eigvals)}")
        
        print("\n[3/5] پروجکشن به فضای PCA...")
        x_rowvector = self.x_train_sample.T
        self.pca_projection = np.dot(x_rowvector, self.pca_eigvecs)
        print(f"      ✓ شکل projection: {self.pca_projection.shape}")
        
        print("\n[4/5] محاسبه واریانس...")
        variance_ratio = self.pca_eigvals / self.pca_eigvals.sum()
        cumulative_var = np.cumsum(variance_ratio)
        
        print(f"      ✓ واریانس PC1: {variance_ratio[0]*100:.2f}%")
        print(f"      ✓ واریانس PC2: {variance_ratio[1]*100:.2f}%")
        print(f"      ✓ واریانس تجمعی 2 مؤلفه: {cumulative_var[1]*100:.2f}%")
        
        print("\n[5/5] محاسبه میانگین هر رقم در فضای PCA...")
        pca_means = {}
        for digit in range(10):
            digit_indices = np.where(self.y_train_sample == digit)[0]
            digit_projection = self.pca_projection[digit_indices, :2]
            pca_means[digit] = np.mean(digit_projection, axis=0)
            print(f"      ✓ رقم {digit}: میانگین = [{pca_means[digit][0]:.2f}, {pca_means[digit][1]:.2f}]")
        
        if visualize:
            self._visualize_pca(pca_means, cumulative_var, save_path)
        
        print("\n" + "═" * 80)
        print("✅ تحلیل PCA با موفقیت انجام شد!")
        print("═" * 80)
        
        return self.pca_eigvals, self.pca_eigvecs, self.pca_projection
    
    def _visualize_pca(self, pca_means: Dict, cumulative_var: np.ndarray,
                      save_path: Optional[str]):
        """ویژوالایزیشن PCA"""
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])
        
        # 1. پراکندگی 2D
        ax1 = plt.subplot(gs[0])
        pc1 = self.pca_projection[:, 0]
        pc2 = self.pca_projection[:, 1]
        
        scatter = ax1.scatter(pc1, pc2, 
                            c=self.y_train_sample,
                            cmap='tab10', alpha=0.5, s=20)
        
        # نمایش میانگین‌ها
        for digit, mean in pca_means.items():
            ax1.scatter(mean[0], mean[1], 
                       c=f'C{digit}', s=200, marker='*',
                       edgecolors='black', linewidths=2,
                       label=f'Mean {digit}')
        
        ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax1.set_title('PCA: نمایش 2D با میانگین هر رقم\nPCA 2D View with Mean Points',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        plt.colorbar(scatter, ax=ax1, label='Digit')
        
        # 2. واریانس تجمعی
        ax2 = plt.subplot(gs[1])
        n_show = min(50, len(cumulative_var))
        ax2.plot(range(1, n_show + 1), 
                cumulative_var[:n_show] * 100,
                'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7,
                   label='90% واریانس')
        ax2.set_xlabel('تعداد مؤلفه\nComponents', fontsize=10)
        ax2.set_ylabel('واریانس تجمعی (%)\nCumulative Variance', fontsize=10)
        ax2.set_title('واریانس تجمعی\nCumulative Variance', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. مقایسه میانگین‌ها
        ax3 = plt.subplot(gs[2])
        digits = list(pca_means.keys())
        pc1_means = [pca_means[d][0] for d in digits]
        pc2_means = [pca_means[d][1] for d in digits]
        
        for i, digit in enumerate(digits):
            ax3.scatter(pc1_means[i], pc2_means[i], 
                       s=300, c=f'C{digit}',
                       marker='$'+str(digit)+'$', 
                       edgecolors='black', linewidths=1.5)
        
        ax3.set_xlabel('PC1 Mean', fontsize=10)
        ax3.set_ylabel('PC2 Mean', fontsize=10)
        ax3.set_title('میانگین PCA برای هر رقم\nPCA Mean per Digit',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      ✓ ذخیره PCA: {save_path}")
        
        plt.show()
    
    def perform_svd(self, visualize: bool = True,
                   save_path: Optional[str] = None) -> Tuple:
        """
        انجام تحلیل SVD روی کل دیتاست
        
        Args:
            visualize: نمایش نتایج
            save_path: مسیر ذخیره
            
        Returns:
            (U, S, Vt)
        """
        print(f"\n{'═'*80}")
        print("🔬 تجزیه مقدار تکین (SVD)")
        print("Singular Value Decomposition")
        print(f"{'═'*80}")
        
        print("\n[1/4] محاسبه SVD...")
        self.svd_U, self.svd_S, self.svd_Vt = np.linalg.svd(
            self.x_train_sample, full_matrices=False
        )
        
        print(f"      ✓ شکل U: {self.svd_U.shape}")
        print(f"      ✓ شکل S: {self.svd_S.shape}")
        print(f"      ✓ شکل Vt: {self.svd_Vt.shape}")
        
        print("\n[2/4] تحلیل مقادیر تکین...")
        print(f"      ✓ بزرگ‌ترین: {self.svd_S[0]:.2f}")
        print(f"      ✓ کوچک‌ترین: {self.svd_S[-1]:.6f}")
        
        # محاسبه انرژی
        energy = (self.svd_S ** 2) / (self.svd_S ** 2).sum()
        cumulative_energy = np.cumsum(energy)
        
        n_90 = np.where(cumulative_energy >= 0.9)[0][0] + 1
        print(f"      ✓ مؤلفه‌های لازم برای 90% انرژی: {n_90}")
        
        print("\n[3/4] پروجکشن به فضای SVD...")
        svd_projection = np.dot(self.svd_U[:, :2].T, self.x_train_sample)
        print(f"      ✓ شکل projection: {svd_projection.shape}")
        
        print("\n[4/4] محاسبه میانگین هر رقم در فضای SVD...")
        svd_means = {}
        for digit in range(10):
            digit_indices = np.where(self.y_train_sample == digit)[0]
            digit_proj = svd_projection[:, digit_indices]
            svd_means[digit] = np.mean(digit_proj, axis=1)
            print(f"      ✓ رقم {digit}: میانگین = [{svd_means[digit][0]:.2f}, {svd_means[digit][1]:.2f}]")
        
        if visualize:
            self._visualize_svd(svd_projection, svd_means, 
                              energy, cumulative_energy, save_path)
        
        print("\n" + "═" * 80)
        print("✅ تحلیل SVD با موفقیت انجام شد!")
        print("═" * 80)
        
        return self.svd_U, self.svd_S, self.svd_Vt
    
    def _visualize_svd(self, projection, svd_means, energy, 
                      cumulative_energy, save_path):
        """ویژوالایزیشن SVD"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        
        # 1. پراکندگی 2D SVD
        ax1 = plt.subplot(gs[0, 0])
        scatter = ax1.scatter(projection[0, :], projection[1, :],
                            c=self.y_train_sample, cmap='tab10',
                            alpha=0.5, s=20)
        
        # میانگین‌ها
        for digit, mean in svd_means.items():
            ax1.scatter(mean[0], mean[1], s=200, marker='*',
                       c=f'C{digit}', edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('SVD Component 1', fontsize=11)
        ax1.set_ylabel('SVD Component 2', fontsize=11)
        ax1.set_title('SVD: نمایش 2D\nSVD 2D Projection', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Digit')
        
        # 2. مقادیر تکین
        ax2 = plt.subplot(gs[0, 1])
        n_plot = min(100, len(self.svd_S))
        ax2.plot(range(1, n_plot + 1), self.svd_S[:n_plot],
                'b-', linewidth=2)
        ax2.set_xlabel('Index', fontsize=10)
        ax2.set_ylabel('Singular Value', fontsize=10)
        ax2.set_title('مقادیر تکین\nSingular Values',
                     fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. انرژی تجمعی
        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(range(1, n_plot + 1), 
                cumulative_energy[:n_plot] * 100,
                'g-', linewidth=2)
        ax3.axhline(y=90, color='r', linestyle='--', label='90%')
        ax3.set_xlabel('Components', fontsize=10)
        ax3.set_ylabel('Cumulative Energy (%)', fontsize=10)
        ax3.set_title('انرژی تجمعی\nCumulative Energy',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4-6: Eigendigits (اولین 3 بردار تکین)
        for i in range(3):
            ax = plt.subplot(gs[1, i])
            eigendigit = self.svd_U[:, i].reshape(28, 28)
            im = ax.imshow(eigendigit, cmap='RdBu_r')
            ax.set_title(f'Eigendigit {i+1}\nبردار تکین {i+1}',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      ✓ ذخیره SVD: {save_path}")
        
        plt.show()
    
    def compute_mean_images(self, visualize: bool = True,
                           save_path: Optional[str] = None):
        """
        محاسبه mean image برای هر رقم
        
        Args:
            visualize: نمایش mean images
            save_path: مسیر ذخیره
        """
        print(f"\n{'═'*80}")
        print("📸 محاسبه Mean Image برای هر رقم")
        print("Computing Mean Image per Digit")
        print(f"{'═'*80}")
        
        self.mean_images = []
        
        print()
        for digit in range(10):
            mean_img = np.mean(self.x_list[digit], axis=1)
            self.mean_images.append(mean_img)
            print(f"  ✓ رقم {digit}: {len(self.y_list[digit])} نمونه")
        
        if visualize:
            self._visualize_mean_images(save_path)
        
        print("\n" + "═" * 80)
        print("✅ Mean Images محاسبه شدند!")
        print("═" * 80)
    
    def _visualize_mean_images(self, save_path):
        """نمایش mean images"""
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        fig.suptitle('تصویر میانگین هر رقم\nMean Image per Digit',
                    fontsize=14, fontweight='bold', y=1.0)
        
        for digit in range(10):
            row = digit // 5
            col = digit % 5
            
            mean_img = self.mean_images[digit].reshape(28, 28)
            axes[row, col].imshow(mean_img, cmap='Greys')
            axes[row, col].set_title(f'Digit {digit}',
                                    fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      ✓ ذخیره mean images: {save_path}")
        
        plt.show()
    
    def classify_with_mean_images(self, n_test: int = 1000) -> Tuple[float, np.ndarray]:
        """
        طبقه‌بندی با استفاده از mean images
        
        Args:
            n_test: تعداد نمونه‌های تست
            
        Returns:
            (accuracy, predictions)
        """
        print(f"\n{'═'*80}")
        print(f"🎯 طبقه‌بندی با Mean Images")
        print(f"Classification using Mean Images")
        print(f"{'═'*80}")
        
        print(f"\n[1/2] پیش‌بینی برای {n_test} نمونه تست...")
        
        predictions = []
        x_test_sample = self.x_test_rowvector[:n_test]
        y_test_sample = self.y_test[:n_test]
        
        for i in range(n_test):
            test_image = x_test_sample[i]
            
            # محاسبه فاصله تا هر mean image
            distances = []
            for digit in range(10):
                dist = np.linalg.norm(test_image - self.mean_images[digit])
                distances.append(dist)
            
            # انتخاب نزدیک‌ترین
            predicted = np.argmin(distances)
            predictions.append(predicted)
        
        predictions = np.array(predictions)
        
        print("\n[2/2] محاسبه دقت...")
        accuracy = np.mean(predictions == y_test_sample)
        
        print(f"\n  ✓ دقت: {accuracy*100:.2f}%")
        print(f"  ✓ تعداد صحیح: {int(accuracy*n_test)}/{n_test}")
        
        print("\n" + "═" * 80)
        print("✅ طبقه‌بندی با Mean Images انجام شد!")
        print("═" * 80)
        
        return accuracy, predictions
    
    def compute_digit_svds(self, visualize: bool = True,
                          save_path: Optional[str] = None):
        """
        محاسبه SVD برای هر رقم به صورت جداگانه
        
        Args:
            visualize: نمایش singular images
            save_path: مسیر ذخیره
        """
        print(f"\n{'═'*80}")
        print("🔍 محاسبه SVD برای هر رقم")
        print("Computing SVD per Digit")
        print(f"{'═'*80}")
        
        print()
        for digit in range(10):
            print(f"  [رقم {digit}] محاسبه SVD...")
            U, S, Vt = np.linalg.svd(self.x_list[digit], full_matrices=False)
            self.digit_svds[digit] = (U, S, Vt)
            print(f"    ✓ U: {U.shape}, S: {S.shape}")
        
        if visualize:
            self._visualize_singular_images(save_path)
        
        print("\n" + "═" * 80)
        print("✅ SVD هر رقم محاسبه شد!")
        print("═" * 80)
    
    def _visualize_singular_images(self, save_path):
        """نمایش 3 singular image اول برای هر رقم"""
        fig, axes = plt.subplots(10, 3, figsize=(10, 28))
        fig.suptitle('سه Singular Image اول برای هر رقم\nTop 3 Singular Images per Digit',
                    fontsize=14, fontweight='bold', y=0.995)
        
        for digit in range(10):
            U = self.digit_svds[digit][0]
            
            for i in range(3):
                singular_img = U[:, i].reshape(28, 28)
                axes[digit, i].imshow(singular_img, cmap='RdBu_r')
                
                if i == 0:
                    axes[digit, i].set_ylabel(f'Digit {digit}',
                                             fontsize=11, fontweight='bold')
                
                if digit == 0:
                    axes[digit, i].set_title(f'Singular #{i+1}',
                                            fontsize=10)
                
                axes[digit, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      ✓ ذخیره singular images: {save_path}")
        
        plt.show()
    
    def classify_with_svd(self, k: int = 5, n_test: int = 1000) -> Tuple[float, np.ndarray]:
        """
        طبقه‌بندی با استفاده از top-k singular vectors
        
        Args:
            k: تعداد singular vectors
            n_test: تعداد نمونه‌های تست
            
        Returns:
            (accuracy, predictions)
        """
        print(f"\n{'═'*80}")
        print(f"🎯 طبقه‌بندی با Top-{k} Singular Vectors")
        print(f"Classification using SVD (k={k})")
        print(f"{'═'*80}")
        
        print(f"\n[1/2] محاسبه projection matrices...")
        
        # محاسبه U_k × U_k^T برای هر رقم
        projection_matrices = []
        for digit in range(10):
            U = self.digit_svds[digit][0]
            U_k = U[:, :k]
            P = np.dot(U_k, U_k.T)
            projection_matrices.append(P)
            print(f"  ✓ رقم {digit}: P shape = {P.shape}")
        
        print(f"\n[2/2] پیش‌بینی برای {n_test} نمونه...")
        
        predictions = []
        x_test_sample = self.x_test_rowvector[:n_test]
        y_test_sample = self.y_test[:n_test]
        
        for i in range(n_test):
            test_image = x_test_sample[i]
            
            # محاسبه فاصله projection برای هر رقم
            distances = []
            for digit in range(10):
                P = projection_matrices[digit]
                projected = np.dot(P, test_image)
                dist = np.linalg.norm(test_image - projected)
                distances.append(dist)
            
            predicted = np.argmin(distances)
            predictions.append(predicted)
        
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test_sample)
        
        print(f"\n  ✓ دقت با k={k}: {accuracy*100:.2f}%")
        print(f"  ✓ تعداد صحیح: {int(accuracy*n_test)}/{n_test}")
        
        print("\n" + "═" * 80)
        print(f"✅ طبقه‌بندی SVD (k={k}) انجام شد!")
        print("═" * 80)
        
        return accuracy, predictions
    
    def test_multiple_k(self, k_values: List[int] = [1, 2, 3, 5, 10],
                       n_test: int = 1000,
                       visualize: bool = True,
                       save_path: Optional[str] = None) -> Dict:
        """
        تست با مقادیر مختلف k و نمایش confusion matrix
        
        Args:
            k_values: لیست مقادیر k
            n_test: تعداد نمونه‌های تست
            visualize: نمایش confusion matrices
            save_path: مسیر ذخیره
            
        Returns:
            دیکشنری نتایج {k: (accuracy, predictions)}
        """
        print(f"\n{'═'*80}")
        print(f"🧪 تست با مقادیر مختلف k")
        print(f"Testing Multiple k Values: {k_values}")
        print(f"{'═'*80}\n")
        
        results = {}
        y_test_sample = self.y_test[:n_test]
        
        for k in k_values:
            print(f"▶ k = {k}")
            accuracy, predictions = self.classify_with_svd(k, n_test)
            results[k] = (accuracy, predictions)
            print()
        
        if visualize:
            self._plot_multiple_confusion_matrices(
                results, y_test_sample, save_path
            )
        
        # نمایش جدول مقایسه
        print("\n" + "═" * 80)
        print("📊 مقایسه نتایج")
        print("═" * 80)
        print(f"{'k':<6} {'Accuracy':<12} {'Correct/Total':<15}")
        print("─" * 80)
        for k in k_values:
            acc = results[k][0]
            correct = int(acc * n_test)
            print(f"{k:<6} {acc*100:>7.2f}%     {correct:>5}/{n_test:<8}")
        print("═" * 80)
        
        return results
    
    def _plot_multiple_confusion_matrices(self, results, y_true, save_path):
        """رسم confusion matrix برای مقادیر مختلف k"""
        n_plots = len(results)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrix برای مقادیر مختلف k\nConfusion Matrices for Different k Values',
                    fontsize=14, fontweight='bold', y=1.05)
        
        for idx, (k, (acc, predictions)) in enumerate(results.items()):
            # محاسبه confusion matrix
            cm = confusion_matrix(y_true, predictions, normalize='true')
            
            # رسم heatmap
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=range(10), yticklabels=range(10),
                       cbar=True, ax=axes[idx])
            
            axes[idx].set_title(f'k = {k}\nAccuracy = {acc*100:.2f}%',
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            
            if idx == 0:
                axes[idx].set_ylabel('True', fontsize=10)
            else:
                axes[idx].set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"      ✓ ذخیره confusion matrices: {save_path}")
        
        plt.show()


def main():
    """تابع اصلی برای اجرای کامل تحلیل"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MNIST Complete SVD & PCA Analysis" + " " * 29 + "║")
    print("║" + " " * 15 + "تحلیل کامل MNIST با SVD و PCA" + " " * 31 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
    print("║" + "  ایمیل: safarireza@gmail.com".ljust(77) + " ║")
    print("║" + "  گیتهاب: https://github.com/reza123reza".ljust(77) + " ║")
    print("╚" + "═" * 78 + "╝\n")
    
    # ایجاد analyzer
    analyzer = MNISTAnalyzer(sample_size=2000)
    
    # 1. بارگذاری داده
    if not analyzer.load_data():
        return
    
    # 2. نمایش نمونه‌ها
    analyzer.visualize_samples(n_samples=10, 
                              save_path='01_mnist_samples.png')
    
    # 3. تحلیل PCA
    analyzer.perform_pca(visualize=True, 
                        save_path='02_pca_analysis.png')
    
    # 4. تحلیل SVD
    analyzer.perform_svd(visualize=True, 
                        save_path='03_svd_analysis.png')
    
    # 5. Mean Images
    analyzer.compute_mean_images(visualize=True,
                                 save_path='04_mean_images.png')
    
    # 6. طبقه‌بندی با Mean Images
    mean_acc, _ = analyzer.classify_with_mean_images(n_test=1000)
    
    # 7. SVD هر رقم
    analyzer.compute_digit_svds(visualize=True,
                                save_path='05_singular_images.png')
    
    # 8. طبقه‌بندی با SVD - مقادیر مختلف k
    results = analyzer.test_multiple_k(
        k_values=[1, 2, 3, 5, 10],
        n_test=1000,
        visualize=True,
        save_path='06_confusion_matrices.png'
    )
    
    # خلاصه نهایی
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 28 + "خلاصه نتایج" + " " * 37 + "║")
    print("║" + " " * 28 + "Results Summary" + " " * 35 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Mean Image Classification: {mean_acc*100:.2f}%".ljust(78) + " ║")
    print("║" + " " * 78 + "║")
    print("║  SVD Classification (k values):".ljust(78) + " ║")
    for k, (acc, _) in results.items():
        print(f"║    k = {k:2d}: {acc*100:.2f}%".ljust(78) + " ║")
    print("╠" + "═" * 78 + "╣")
    print("║" + " " * 25 + "✅ تحلیل کامل انجام شد!" + " " * 26 + "║")
    print("║" + " " * 78 + "║")
    print("║  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
    print("║  با تشکر از استفاده".ljust(77) + " ║")
    print("╚" + "═" * 78 + "╝\n")


if __name__ == "__main__":
    """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
    ║  ایمیل: safarireza@gmail.com                                              ║
    ║  گیتهاب: https://github.com/reza123reza                                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    main()
