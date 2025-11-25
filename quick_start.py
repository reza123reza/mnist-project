"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        Quick Start Example                                 ║
║                      مثال شروع سریع                                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║  نویسنده: رضا صفری فروشانی (Reza Safari Froushani)                      ║
║  ایمیل: safarireza@gmail.com                                              ║
║  گیتهاب: https://github.com/reza123reza                                   ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from mnist_svd_complete import MNISTAnalyzer
import numpy as np

def quick_demo():
    """
    دموی سریع با نمایش قابلیت‌های اصلی
    Quick demo showing main capabilities
    """
    print("\n" + "="*80)
    print("🚀 شروع سریع MNIST SVD Analysis")
    print("Quick Start MNIST SVD Analysis")
    print("="*80)
    
    # ایجاد analyzer با نمونه کم برای سرعت بیشتر
    print("\n[1/8] ساخت analyzer...")
    analyzer = MNISTAnalyzer(sample_size=1000)
    
    # بارگذاری داده
    print("\n[2/8] بارگذاری MNIST...")
    if not analyzer.load_data():
        print("❌ خطا در بارگذاری!")
        return
    
    # نمایش چند نمونه
    print("\n[3/8] نمایش نمونه‌ها...")
    analyzer.visualize_samples(n_samples=10, save_path='quick_samples.png')
    
    # تحلیل PCA
    print("\n[4/8] تحلیل PCA...")
    eigenvals, eigenvecs, _ = analyzer.perform_pca(visualize=True,
                                                    save_path='quick_pca.png')
    
    # تحلیل SVD
    print("\n[5/8] تحلیل SVD...")
    U, S, Vt = analyzer.perform_svd(visualize=True,
                                     save_path='quick_svd.png')
    
    # Mean Images
    print("\n[6/8] محاسبه Mean Images...")
    analyzer.compute_mean_images(visualize=True,
                                  save_path='quick_means.png')
    
    # طبقه‌بندی با Mean Images
    print("\n[7/8] طبقه‌بندی با Mean Images...")
    mean_acc, _ = analyzer.classify_with_mean_images(n_test=500)
    
    # طبقه‌بندی با SVD
    print("\n[8/8] طبقه‌بندی با SVD (k=3)...")
    analyzer.compute_digit_svds(visualize=False)
    svd_acc, _ = analyzer.classify_with_svd(k=3, n_test=500)
    
    # نمایش نتایج نهایی
    print("\n" + "="*80)
    print("📊 نتایج نهایی / Final Results")
    print("="*80)
    print(f"  Mean Image Classification: {mean_acc*100:.2f}%")
    print(f"  SVD Classification (k=3):  {svd_acc*100:.2f}%")
    print("="*80)
    
    print("\n✅ دمو با موفقیت انجام شد!")
    print("✅ Demo completed successfully!")
    print("\n💾 فایل‌های ذخیره شده:")
    print("   - quick_samples.png")
    print("   - quick_pca.png")
    print("   - quick_svd.png")
    print("   - quick_means.png")
    

def minimal_example():
    """
    کوچک‌ترین مثال ممکن
    Minimal example
    """
    print("\n" + "="*80)
    print("⚡ مثال مینیمال / Minimal Example")
    print("="*80)
    
    # فقط یک تحلیل ساده
    analyzer = MNISTAnalyzer(sample_size=500)
    analyzer.load_data()
    
    # فقط Mean Image Classification
    analyzer.compute_mean_images(visualize=False)
    accuracy, _ = analyzer.classify_with_mean_images(n_test=200)
    
    print(f"\n✅ دقت: {accuracy*100:.2f}%")


def compare_methods():
    """
    مقایسه روش‌های مختلف
    Compare different methods
    """
    print("\n" + "="*80)
    print("🔬 مقایسه روش‌های مختلف")
    print("Comparing Different Methods")
    print("="*80)
    
    analyzer = MNISTAnalyzer(sample_size=1500)
    analyzer.load_data()
    
    # روش 1: Mean Image
    print("\n[Method 1] Mean Image Classification...")
    analyzer.compute_mean_images(visualize=False)
    mean_acc, _ = analyzer.classify_with_mean_images(n_test=500)
    
    # روش 2: SVD با k مختلف
    print("\n[Method 2] SVD Classification...")
    analyzer.compute_digit_svds(visualize=False)
    
    k_values = [1, 3, 5, 10]
    svd_results = {}
    
    for k in k_values:
        acc, _ = analyzer.classify_with_svd(k=k, n_test=500)
        svd_results[k] = acc
        print(f"  k={k}: {acc*100:.2f}%")
    
    # نمایش مقایسه
    print("\n" + "="*80)
    print("📊 جدول مقایسه / Comparison Table")
    print("="*80)
    print(f"{'Method':<25} {'Accuracy':<12}")
    print("-"*80)
    print(f"{'Mean Image':<25} {mean_acc*100:>7.2f}%")
    for k, acc in svd_results.items():
        print(f"{'SVD (k=' + str(k) + ')':<25} {acc*100:>7.2f}%")
    print("="*80)
    
    # تعیین بهترین روش
    best_k = max(svd_results, key=svd_results.get)
    best_acc = svd_results[best_k]
    
    if best_acc > mean_acc:
        print(f"\n🏆 بهترین روش: SVD با k={best_k} ({best_acc*100:.2f}%)")
    else:
        print(f"\n🏆 بهترین روش: Mean Image ({mean_acc*100:.2f}%)")


def custom_analysis():
    """
    تحلیل سفارشی - فقط یک رقم خاص
    Custom analysis - only one specific digit
    """
    print("\n" + "="*80)
    print("🎯 تحلیل سفارشی - رقم 7")
    print("Custom Analysis - Digit 7")
    print("="*80)
    
    analyzer = MNISTAnalyzer(sample_size=2000)
    analyzer.load_data()
    
    # فقط رقم 7
    digit = 7
    digit_images = analyzer.x_list[digit]
    digit_labels = analyzer.y_list[digit]
    
    print(f"\nتعداد نمونه رقم {digit}: {len(digit_labels)}")
    
    # SVD رقم 7
    print("محاسبه SVD...")
    U, S, Vt = np.linalg.svd(digit_images, full_matrices=False)
    
    # تحلیل مقادیر تکین
    energy = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(energy)
    
    print(f"\nانرژی 5 مقدار تکین اول:")
    for i in range(5):
        print(f"  Singular Value {i+1}: {energy[i]*100:.2f}%")
    
    print(f"\nانرژی تجمعی 5 مؤلفه اول: {cumulative[4]*100:.2f}%")
    print(f"انرژی تجمعی 10 مؤلفه اول: {cumulative[9]*100:.2f}%")
    
    # نمایش Eigendigits
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.suptitle(f'پنج Eigendigit اول برای رقم {digit}\n' +
                f'First 5 Eigendigits for Digit {digit}',
                fontsize=12, fontweight='bold')
    
    for i in range(5):
        eigendigit = U[:, i].reshape(28, 28)
        axes[i].imshow(eigendigit, cmap='RdBu_r')
        axes[i].set_title(f'#{i+1}\n{energy[i]*100:.1f}%')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'digit_{digit}_eigendigits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 ذخیره شد: digit_{digit}_eigendigits.png")


def menu():
    """منوی انتخاب مثال"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "انتخاب مثال / Choose Example" + " "*28 + "║")
    print("╠" + "═"*78 + "╣")
    print("║  1. دموی سریع / Quick Demo (توصیه می‌شود)".ljust(77) + " ║")
    print("║  2. مثال مینیمال / Minimal Example (سریع‌ترین)".ljust(77) + " ║")
    print("║  3. مقایسه روش‌ها / Compare Methods".ljust(77) + " ║")
    print("║  4. تحلیل سفارشی / Custom Analysis (رقم 7)".ljust(77) + " ║")
    print("║  5. خروج / Exit".ljust(77) + " ║")
    print("╚" + "═"*78 + "╝")
    
    choice = input("\nانتخاب شما (1-5): ")
    
    if choice == '1':
        quick_demo()
    elif choice == '2':
        minimal_example()
    elif choice == '3':
        compare_methods()
    elif choice == '4':
        custom_analysis()
    elif choice == '5':
        print("\n👋 خداحافظ!")
        return
    else:
        print("\n❌ انتخاب نامعتبر!")
        return
    
    # پرسیدن برای ادامه
    again = input("\n\nاجرای مثال دیگر؟ (y/n): ")
    if again.lower() == 'y':
        menu()


if __name__ == "__main__":
    """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  نویسنده: رضا صفری فروشانی                                                ║
    ║  Author: Reza Safari Froushani                                             ║
    ║  Email: safarireza@gmail.com                                               ║
    ║  GitHub: https://github.com/reza123reza                                    ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║  Copyright © 2024 Reza Safari Froushani. All Rights Reserved.             ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*15 + "MNIST SVD Analysis - Quick Start Examples" + " "*21 + "║")
    print("║" + " "*78 + "║")
    print("║" + "  نویسنده: رضا صفری فروشانی".ljust(77) + " ║")
    print("║" + "  ایمیل: safarireza@gmail.com".ljust(77) + " ║")
    print("║" + "  گیتهاب: https://github.com/reza123reza".ljust(77) + " ║")
    print("╚" + "═"*78 + "╝")
    
    # نمایش منو
    menu()
