import matplotlib.pyplot as plt
import plotly.express as px
import cv2


def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def visualize_prediction(image_path, prediction, confidence, save_path=None):
    """可视化预测结果"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f'Prediction: {prediction}\nConfidence: {confidence:.2f}')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names,
        y=class_names,
        title="Confusion Matrix"
    )

    if save_path:
        fig.write_image(save_path)

    fig.show()


def plot_heatmap(heatmap, save_path=None):
    """绘制热力图"""
    fig = px.imshow(
        heatmap,
        color_continuous_scale='hot',
        title="Abnormal Cell Distribution Heatmap"
    )

    if save_path:
        fig.write_image(save_path)

    fig.show()