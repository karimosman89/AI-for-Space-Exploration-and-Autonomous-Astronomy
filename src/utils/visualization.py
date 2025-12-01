"""
Visualization utilities for results, training curves, and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import seaborn as sns


def visualize_results(
    images: List[np.ndarray],
    predictions: List[Dict],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize prediction results on images.
    
    Args:
        images: List of input images
        predictions: List of prediction dictionaries
        titles: Optional titles for each subplot
        save_path: Optional path to save figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, pred) in enumerate(zip(images, predictions)):
        axes[i].imshow(img)
        axes[i].axis('off')
        
        title = titles[i] if titles else f"{pred.get('class', 'N/A')}\nConf: {pred.get('confidence', 0):.2f}"
        axes[i].set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12, 4))
    
    if not train_accs:
        axes = [axes]
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies if provided
    if train_accs:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_detection_results(
    image: np.ndarray,
    detections: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot object detection results with bounding boxes.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(image)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for det in detections:
        bbox = det['bbox']
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        
        # Get color based on class
        class_name = det['class']
        color = colors[hash(class_name) % 10]
        
        # Draw box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=3
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name}: {det['confidence']:.2f}"
        ax.text(
            x1, y1 - 5,
            label,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax.axis('off')
    ax.set_title('Detection Results', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_trajectory(
    states: List[np.ndarray],
    goal: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot 3D trajectory of spacecraft navigation.
    
    Args:
        states: List of state vectors (containing position)
        goal: Optional goal position
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = np.array([state[:3] for state in states])
    
    # Plot trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        'b-',
        linewidth=2,
        label='Trajectory'
    )
    
    # Mark start and end
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        c='green',
        s=200,
        marker='o',
        label='Start'
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        c='red',
        s=200,
        marker='x',
        label='End'
    )
    
    # Plot goal if provided
    if goal is not None:
        ax.scatter(
            goal[0],
            goal[1],
            goal[2],
            c='gold',
            s=300,
            marker='*',
            label='Goal'
        )
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Z Position', fontsize=12)
    ax.set_title('Spacecraft Trajectory', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
