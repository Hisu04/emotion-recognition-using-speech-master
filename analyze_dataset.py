import matplotlib.pyplot as plt
import pandas as pd
from emotion_recognition import EmotionRecognizer
import numpy as np
import sys

# Hỗ trợ in tiếng Việt trên terminal Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def analyze_and_visualize():
    print("[*] Đang phân tích tập dữ liệu...")
    
    # Khởi tạo recognizer để load data
    emotions = ["sad", "neutral", "happy", "angry", "fear", "disgust", "ps", "boredom"]
    rec = EmotionRecognizer(emotions=emotions, verbose=0)
    
    # Lấy thông tin số lượng mẫu
    # get_samples_by_class trả về DataFrame có cột 'train', 'test', 'total'
    df = rec.get_samples_by_class()
    
    # Loại bỏ hàng 'total' cuối cùng để vẽ biểu đồ
    plot_df = df.iloc[:-1]
    
    # 1. Vẽ biểu đồ phân bổ lớp (Class Distribution)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_df.index))
    width = 0.35
    
    plt.bar(x - width/2, plot_df['train'], width, label='Training', color='#3498db')
    plt.bar(x + width/2, plot_df['test'], width, label='Testing', color='#e67e22')
    
    plt.xlabel('Cảm xúc (Emotions)')
    plt.ylabel('Số lượng mẫu (Samples)')
    plt.title('Phân bổ dữ liệu theo từng loại cảm xúc (Dataset Distribution)')
    plt.xticks(x, plot_df.index)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight sự mất cân bằng
    max_val = plot_df['train'].max()
    min_val = plot_df['train'].min()
    print(f"[!] Nhận xét: Lớp nhiều nhất ({plot_df['train'].idxmax()}: {max_val}) so với lớp ít nhất ({plot_df['train'].idxmin()}: {min_val})")
    
    save_path = "dataset_distribution.png"
    plt.savefig(save_path)
    print(f"[+] Đã lưu biểu đồ phân bổ tại: {save_path}")
    
    # 2. Vẽ Confusion Matrix (Mô phỏng hoặc thực tế sau khi train)
    print("[*] Đang tạo Ma trận nhầm lẫn (Confusion Matrix)...")
    try:
        rec.train()
        matrix_df = rec.confusion_matrix(percentage=True, labeled=True)
        
        plt.figure(figsize=(10, 8))
        import seaborn as sns # Kiểm tra xem có seaborn không, nếu không dùng matplotlib thuần
        sns.heatmap(matrix_df, annot=True, fmt=".1f", cmap="Blues")
        plt.title("Ma trận nhầm lẫn (Confusion Matrix Heatmap %)")
        plt.savefig("confusion_matrix_heatmap.png")
        print("[+] Đ đã lưu Confusion Matrix Heatmap tại: confusion_matrix_heatmap.png")
    except Exception as e:
        print(f"[!] Không thể vẽ Confusion Matrix thực tế (cần train xong): {e}")

    plt.show()

if __name__ == "__main__":
    analyze_and_visualize()
